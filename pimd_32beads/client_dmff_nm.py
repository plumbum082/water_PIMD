#!/usr/bin/env python3
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import sys
import driver
import numpy as np
import jax
import jax.numpy as jnp
import dmff.admp.pme
from intra import onebodyenergy
from jax_md import space, partition
from jax import jit, vmap, value_and_grad
from dmff.utils import jit_condition
import openmm.app as app
import openmm.unit as unit
dmff.PRECISION = 'float'
dmff.update_jax_precision(dmff.PRECISION)
from dmff.api import Hamiltonian
import pickle
from dmff.admp.pme import trim_val_0
from dmff.admp.spatial import v_pbc_shift
# from dmff.common import nblist
import nblist
from dmff.admp.pairwise import (
    TT_damping_qq_c6_kernel,
    generate_pairwise_interaction,
    slater_disp_damping_kernel,
    slater_sr_kernel,
    TT_damping_qq_kernel
)
from jax import config
config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", False)
from dmff.eann.eann import EANNForce, get_elem_indices
import dmff
from dmff.admp import pme
pme.DEFAULT_THOLE_WIDTH = 2.6
#compute geometry dependent charge/dispersion
@jit_condition(static_argnums=())
def compute_leading_terms(positions,box):
    n_atoms = len(positions)
    c0 = jnp.zeros(n_atoms)
    c6_list = jnp.zeros(n_atoms)
    box_inv = jnp.linalg.inv(box)
    O = positions[::3]
    H1 = positions[1::3]
    H2 = positions[2::3]
    ROH1 = H1 - O
    ROH2 = H2 - O
    ROH1 = v_pbc_shift(ROH1, box, box_inv)
    ROH2 = v_pbc_shift(ROH2, box, box_inv)
    # compute bond length and bond angle
    dROH1 = jnp.linalg.norm(ROH1, axis=1)
    dROH2 = jnp.linalg.norm(ROH2, axis=1)
    costh = jnp.sum(ROH1 * ROH2, axis=1) / (dROH1 * dROH2)
    angle = jnp.arccos(costh)*180/jnp.pi
    # compute charge
    dipole1 = -0.016858755+0.002287251*angle + 0.239667591*dROH1 + (-0.070483437)*dROH2
    charge_H1 = dipole1/dROH1
    dipole2 = -0.016858755+0.002287251*angle + 0.239667591*dROH2 + (-0.070483437)*dROH1
    charge_H2 = dipole2/dROH2
    charge_O = -(charge_H1 + charge_H2)
    # compute C6
    C6_H1 = (-2.36066199 + (-0.007049238)*angle + 1.949429648*dROH1+ 2.097120784*dROH2) * 0.529**6 * 2625.5
    C6_H2 = (-2.36066199 + (-0.007049238)*angle + 1.949429648*dROH2+ 2.097120784*dROH1) * 0.529**6 * 2625.5
    C6_O = (-8.641301261 + 0.093247893*angle + 11.90395358*(dROH1+ dROH2)) * 0.529**6 * 2625.5
    C6_H1 = trim_val_0(C6_H1)
    C6_H2 = trim_val_0(C6_H2)
    c0 = c0.at[::3].set(charge_O)
    c0 = c0.at[1::3].set(charge_H1)
    c0 = c0.at[2::3].set(charge_H2)
    c6_list = c6_list.at[::3].set(jnp.sqrt(C6_O))
    c6_list = c6_list.at[1::3].set(jnp.sqrt(C6_H1))
    c6_list = c6_list.at[2::3].set(jnp.sqrt(C6_H2))
    return c0, c6_list


#compute isotropic short-range/Tang Tonnies damping for charge-charge interaction/C6,C8,C10 damping for dispersion
@vmap
@jit
def TT_damping_qq_disp_kernel(dr, m, ai, aj, bi, bj, qi, qj, c6i, c6j, c8i, c8j, c10i, c10j):
    a = jnp.sqrt(ai * aj)
    b = jnp.sqrt(bi * bj)
    c6 = c6i * c6j
    c8 = c8i * c8j
    c10 = c10i * c10j
    q = qi * qj
    r = dr * 1.889726878 # convert to bohr
    br = b * r
    br2 = br * br
    br3 = br2 * br
    br4 = br2 * br2
    br5 = br3 * br2
    br6 = br3 * br3
    br7 = br3 * br4
    br8 = br4 * br4
    br9 = br4 * br5
    br10 = br5 * br5
    exp_br = jnp.exp(-br)
    f = 2625.5 * a * exp_br \
        + (-2625.5) * exp_br * (1+br) * q / r \
        + exp_br*(1+br+br2/2+br3/6+br4/24+br5/120+br6/720) * c6 / dr**6 \
        + exp_br*(1+br+br2/2+br3/6+br4/24+br5/120+br6/720+br7/5040+br8/40320) * c8 / dr**8 \
        + exp_br*(1+br+br2/2+br3/6+br4/24+br5/120+br6/720+br7/5040+br8/40320+br9/362880+br10/3628800) * c10 / dr**10

    return f * m


class DMFFDriver(driver.BaseDriver):

    def __init__(self, addr, port, pdb, f_xml, r_xml, socktype, device='cpu'):
        addr = addr + '_%s'%os.environ['SLURM_JOB_ID']
        # set up the interface with ipi
        driver.BaseDriver.__init__(self, port, addr, socktype)

        # set up various force calculators
        H = Hamiltonian(f_xml)
        app.Topology.loadBondDefinitions(r_xml)
        pdb = app.PDBFile(pdb)
        disp_generator, pme_generator, EANN_generator = H.getGenerators()
        rc = 0.8
        # generator stores all force field parameters    
        pots = H.createPotential(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.nanometer, step_pol=5)
        pot_disp = pots.dmff_potentials['ADMPDispForce']
        pot_pme = pots.dmff_potentials['ADMPPmeForce']
        TT_damping_qq_disp = generate_pairwise_interaction(TT_damping_qq_disp_kernel, static_args={})
        pot_EANN = pots.dmff_potentials['EANNForce']

        #load params
        paramset = H.getParameters()
        params_pme = paramset['ADMPPmeForce']
        params_disp = paramset['ADMPDispForce'] 
        params_eann = paramset['EANNForce']

        # construct inputs
        positions = jnp.array(pdb.positions._value)
        a, b, c = pdb.topology.getPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value])
        L = box[0][0]
        # params = {
        #         'pos': positions,
        #         'box': box,
        #         }
        # with open('pos_box_0.pickle', 'wb') as f:
        #     pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

        # neighbor list
        # self.nbl = nblist.NeighborListFreud(box, rc, pots.meta["cov_map"])
        # self.nbl.allocate(positions)
        # pairs = self.nbl.pairs
        cov_map = pots.meta["cov_map"]
        nbl = nblist.NeighborList(box, rc, cov_map)
        nbl.allocate(positions)

        map_atomtype = pots.meta["ADMPPmeForce_map_atomtype"]
        map_poltype = pots.meta["ADMPPmeForce_map_poltype"]
        map_atype_disp = pots.meta["ADMPDispForce_map_atomtype"]

        def admp_calculator(positions, L):
            box = jnp.array([[L,0,0],[0,L,0],[0,0,L]])          

            # newnblist
            nblist = nbl.nblist.update(positions, box=box)
            pairs_0 = nblist.idx.T
            nbond = cov_map[pairs_0[:, 0], pairs_0[:, 1]]
            pairs = jnp.concatenate([pairs_0, nbond[:, None]], axis=1)
            
            positions = positions *10
            box = box * 10
            # params = {
            #         'pos': positions,
            #         'box': box,
            #         }
            # with open('pos_box_0.pickle', 'wb') as f:
            #     pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            c0, c6_list = compute_leading_terms(positions,box) # compute fluctuated leading terms
            Q_local = params_pme["Q_local"][map_atomtype]
            Q_local = Q_local.at[:,0].set(c0)  # change fixed charge into fluctuated one
            pol = params_pme["pol"][map_poltype]
            tholes = params_pme["thole"][map_poltype]
            c8_list = jnp.sqrt(params_disp["C8"][map_atype_disp]*1e8)
            c10_list = jnp.sqrt(params_disp["C10"][map_atype_disp]*1e10)
            c_list = jnp.vstack((c6_list, c8_list, c10_list))
            a_list = (params_disp["A"][map_atype_disp] / 2625.5)
            b_list = params_disp["B"][map_atype_disp] * 0.0529177249 
            q_list = params_disp["Q"][map_atype_disp]
            E_pme = pme_generator.pme_force.get_energy(
                    positions, box, pairs, Q_local, pol, tholes, pme_generator.mScales, pme_generator.pScales, pme_generator.dScales
                    )    
            E_disp = disp_generator.disp_pme_force.get_energy(positions, box, pairs, c_list.T, disp_generator.mScales)
            E_sr = TT_damping_qq_disp(positions, box, pairs, disp_generator.mScales, a_list, b_list,  c0, c_list[0], c_list[1], c_list[2])
            
            E_intra = onebodyenergy(positions, box)  # compute intramolecular energy     

            n_elem, elem_indices = get_elem_indices(pdb.topology)
            mlmodel = EANNForce(n_elem, elem_indices, n_gto=12, nipsin=2, rc=4)
            E_eann = mlmodel.get_energy(positions, box, pairs, params_eann)
            
            return E_pme - E_disp + E_sr + E_intra + E_eann

        self.tot_force = jit(jax.value_and_grad(admp_calculator,argnums=(0,1)))

        # compile tot_force function
        energy, (grad, virial) = self.tot_force(positions, L)
        print('energy0', energy)
        # print(grad)
        # print(virial)
        return

    def grad(self, crd, cell): # receive SI input, return SI values
        positions = jnp.array(crd*1e9) # convert to nanometer
        box = jnp.array(cell*1e9)      # convert to nanometer
        L = box[0][0]        
        # params = {
        #         'pos': positions,
        #         'box': box,
        #         }
        # with open('pos_box_0.pickle', 'wb') as f:
        #     pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL) 

        # nb list
        # self.nbl.update(positions, box)
        # pairs = self.nbl.pairs
        # energy, (grad, virial) = self.tot_force(positions, L, pairs)
        energy, (grad, virial) = self.tot_force(positions, L)
        virial = np.diag((-grad * positions).sum(axis=0) - virial*L/3).ravel()

        # convert to SI
        energy = np.array(energy * 1000 / 6.0221409e+23) # kj/mol to Joules
        grad = np.array(grad * 1000 / 6.0221409e+23 * 1e9) # convert kj/mol/nm to joule/m
        virial = np.array(virial * 1000 / 6.0221409e+23) # kj/mol to Joules
        return energy, grad, virial


if __name__ == '__main__':
    # the forces are composed by three parts: 
    # the long range part computed using openmm, parameters in xml
    # the short range part writen by hand, parameters in psr
    fn_pdb = sys.argv[1] # pdb file used to define openmm topology, this one should contain all virtual sites
    f_xml = sys.argv[2] # xml file that defines the force field
    r_xml = sys.argv[3] # xml file that defines residues
    addr = sys.argv[4]
    port = int(sys.argv[5])
    socktype = sys.argv[6] 
    driver_dmff = DMFFDriver(addr, port, fn_pdb, f_xml, r_xml, socktype)
    while True:
        driver_dmff.parse()


