# Electrostatic Complementarity in Enzyme Active Sites

## Theory and Physical Basis for Poisson-Boltzmann Electrostatic Complementarity Scoring

---

## 1. Starting Point: Coulomb's Law in a Vacuum

Everything begins with the fact that charges exert forces on each other. If you have a charge $q_1$ sitting at some point and you want to know the electrostatic potential (think of it as "voltage") it creates at a distance $r$, it's:

$$\phi = \frac{q_1}{4\pi\epsilon_0 r}$$

This potential tells you the energy a test charge would experience if you placed there. For a protein with thousands of partial charges (every atom carries a partial charge from the asymmetric sharing of electrons in bonds), the total potential at any point is the sum of contributions from every atom. Conceptually simple, but the problem is that this equation assumes everything is happening in a vacuum, and your enzyme is decidedly not in a vacuum.

---

## 2. The Dielectric Problem

Biology makes physics complicated here. An enzyme has at least two very different electrostatic environments:

- **The protein interior** is mostly hydrocarbon-like — tightly packed sidechains, backbone atoms, relatively few water molecules.
- **The surrounding solvent** is water, which is spectacularly good at screening (weakening) electrostatic interactions.

The physical reason for this difference is that water molecules are permanent dipoles. When you place a charge in water, nearby water molecules reorient their dipoles to partially cancel the field from that charge. This reorientation is captured by the **dielectric constant** $\epsilon$:

| Environment      | Dielectric constant ($\epsilon$) | Physical basis                                               |
| ---------------- | -------------------------------- | ------------------------------------------------------------ |
| Vacuum           | 1                                | No screening                                                 |
| Protein interior | ~2–4                             | Electronic polarization, small backbone fluctuations         |
| Bulk water       | ~80                              | Freely rotating permanent dipoles, electronic polarization   |

Electrostatic interactions in water are roughly 20–40 times weaker than in the protein core.

**This is the fundamental reason enzymes work as electrostatic catalysts.** By placing charged and polar groups in a low-dielectric environment that's been evolutionarily optimized, the enzyme creates much stronger electrostatic effects on the substrate than water ever could. The low dielectric amplifies the impact of every catalytic residue's charge. Water would screen those same interactions into insignificance.

But now you have a computational problem: you can't use one dielectric constant for the whole system. You need to solve for the potential in a system where $\epsilon$ varies spatially — low inside the protein and high outside.

---

## 3. The Poisson Equation

The equation that governs electrostatics in a medium with spatially varying dielectric is the Poisson equation:

$$\nabla \cdot [\epsilon(\mathbf{r}) \nabla \phi(\mathbf{r})] = -\frac{\rho(\mathbf{r})}{\epsilon_0}$$

### Term-by-term breakdown

- $\phi(\mathbf{r})$ — the electrostatic potential at every point in space. **This is what we solve for.**
- $\rho(\mathbf{r})$ — the charge density, i.e., where the charges are. For a protein, this is a collection of point charges (partial charges on each atom from the force field), so $\rho$ is zero almost everywhere except at atom centers.
- $\epsilon(\mathbf{r})$ — the dielectric constant at each point. Low inside the protein, high in solvent.
- $\nabla \phi$ — the gradient of the potential, which is the electric field.
- $\nabla \cdot [\epsilon \nabla \phi]$ — the divergence of the dielectric-weighted electric field, called the **electric displacement**.

**Physical meaning:** the sources of electrostatic potential are the charges ($\rho$), but the way that potential spreads through space depends on the local dielectric ($\epsilon$). Where $\epsilon$ is high (solvent), the potential decays faster. Where $\epsilon$ is low (protein interior), the potential reaches further and is stronger.

If $\epsilon$ were the same everywhere, this would collapse back to the simple Coulomb sum. The entire motivation for solving Poisson is to handle the boundary between protein and solvent correctly.

---

## 4. Adding Salt: The Poisson-Boltzmann Extension

Enzymes operate in buffer with dissolved ions (Na⁺, K⁺, Cl⁻, etc.). These mobile ions also screen electrostatic interactions — positive ions accumulate near negative protein charges and vice versa, partially neutralizing them. This is **Debye-Hückel screening**.

The Boltzmann distribution tells us the concentration of ions at any point:

$$c_\pm(\mathbf{r}) = c_0 \exp\left(\mp \frac{e\phi(\mathbf{r})}{k_BT}\right)$$

where:
- $c_0$ is the bulk ion concentration
- $e$ is the elementary charge
- $k_B T$ is thermal energy

Positive ions are enriched where the potential is negative, and depleted where it's positive.

Substituting this ion distribution into the Poisson equation as an additional charge density gives the full **Poisson-Boltzmann equation**:

$$\nabla \cdot [\epsilon(\mathbf{r}) \nabla \phi(\mathbf{r})] - \kappa^2(\mathbf{r}) \sinh\left(\frac{e\phi}{k_BT}\right) = -\frac{\rho_{\text{fixed}}(\mathbf{r})}{\epsilon_0}$$

where $\kappa$ is the **inverse Debye length**, related to ionic strength by:

$$\kappa^2 = \frac{2c_0 e^2}{\epsilon_s k_B T}$$

At physiological ionic strength (~150 mM), the Debye length is about **8 Å**, meaning ion screening becomes significant beyond roughly that distance.

### Linearization

The $\sinh$ term makes this equation nonlinear and harder to solve. For most protein applications, the potentials aren't large enough to make the nonlinearity matter much, so we use the **linearized Poisson-Boltzmann** (LPB) equation:

$$\nabla \cdot [\epsilon(\mathbf{r}) \nabla \phi(\mathbf{r})] - \kappa^2(\mathbf{r})\phi(\mathbf{r}) = -\frac{\rho_{\text{fixed}}(\mathbf{r})}{\epsilon_0}$$

where $\sinh(x) \approx x$ for small $x$. **This is what APBS and most tools actually solve.** The approximation breaks down only very close to highly charged groups (like a cluster of four glutamates), and even then the error in comparative scoring across variants is small.

Note that $\kappa^2(\mathbf{r})$ is **zero inside the protein** (ions can't penetrate the protein interior) and takes its bulk value in solvent. This is another spatial boundary the solver must handle.

---

## 5. How It's Actually Solved: The Grid

This equation cannot be solved analytically for a shape as complex as a protein. Instead, it is solved numerically by:

1. **Discretizing** space into a 3D grid (typically 0.5–1.0 Å spacing).
2. **Assigning** $\epsilon$ and $\kappa$ values to each grid point based on whether it's inside the protein or in solvent.
3. **Placing** the atomic charges on nearby grid points (charge interpolation).
4. **Solving** the resulting linear system iteratively (usually by multigrid or finite difference methods) until the potential converges.

### The molecular surface

The boundary between protein interior ($\epsilon = 2$–$4$) and solvent ($\epsilon = 80$) is typically defined as the **solvent-excluded surface** (SES): the surface traced by rolling a 1.4 Å sphere (water probe) over the van der Waals surface of the protein. Everything inside the SES gets the protein dielectric; everything outside gets the solvent dielectric.

### Output

The output is $\phi(\mathbf{r})$ at every grid point — a complete **3D map of electrostatic potential** throughout and around the protein.

---

## 6. From Potential Maps to Electrostatic Complementarity

The central idea is straightforward: if the protein is a good electrostatic partner for the ligand, then wherever the ligand has a positive partial charge, the protein should be creating a negative potential, and vice versa.

### Procedure

**Step 1.** Take the protein–ligand complex structure. Remove the ligand. Solve PB for the **protein alone**. This gives $\phi_{\text{protein}}(\mathbf{r})$ — the potential the protein creates in the empty binding site.

**Step 2.** Remove the protein. Solve PB for the **ligand alone**. This gives $\phi_{\text{ligand}}(\mathbf{r})$.

**Step 3.** Define a set of **surface points** at the protein–ligand interface. These are typically points on the molecular surface of the ligand that are in contact with the protein (within some distance cutoff, or the ligand's solvent-accessible surface points that become buried upon complexation).

**Step 4.** At each surface point $i$, evaluate both $\phi_{\text{protein},i}$ and $\phi_{\text{ligand},i}$.

**Step 5.** Compute the **Pearson correlation**:

$$EC = \frac{\sum_i (\phi_{\text{protein},i} - \bar{\phi}_{\text{protein}})(\phi_{\text{ligand},i} - \bar{\phi}_{\text{ligand}})}{n \cdot \sigma_{\text{protein}} \cdot \sigma_{\text{ligand}}}$$

### Interpreting the EC score

| EC value | Interpretation                                              |
| -------- | ----------------------------------------------------------- |
| –1.0     | Perfect complementarity (ideal, never observed in practice) |
| –0.4 to –0.8 | Typical range for well-optimized enzyme active sites   |
| ~0       | No electrostatic relationship                               |
| > 0      | Repulsive electrostatics (unstable complex)                 |

Mutations that disrupt catalysis frequently reduce the magnitude of EC.

---

## 7. Why PB-Based EC Is More Informative Than Simple Coulomb

The PB-based EC captures several things a simple Coulomb sum misses:

### 7.1 Correct treatment of the dielectric boundary

A charged residue buried in the protein interior has a much larger electrostatic effect than the same residue on the solvent-exposed surface. PB captures this; simple Coulomb with a uniform dielectric does not.

### 7.2 Reaction field effects

When you place a charge inside a low-dielectric cavity surrounded by high-dielectric solvent, the solvent polarization creates an additional **reaction field** that modifies the potential inside the cavity. This is physically real and quantitatively significant. It arises naturally from the PB solution.

### 7.3 Ionic screening

Interactions beyond ~8 Å at physiological salt are significantly attenuated. The PB framework handles this naturally through the $\kappa^2 \phi$ term.

### 7.4 Spatial resolution

Rather than just "is the total electrostatic energy favorable," the PB-EC gives you a map of **where specifically** complementarity is good or bad. This lets you identify which residues to target for mutagenesis.

---

## 8. The Solvation Energy Decomposition (Complementary Metric)

In addition to the correlation-based EC, thermodynamic quantities can be extracted. The electrostatic contribution to binding free energy is:

$$\Delta G_{\text{elec}} = G_{\text{elec}}^{\text{complex}} - G_{\text{elec}}^{\text{protein}} - G_{\text{elec}}^{\text{ligand}}$$

Each term is the total electrostatic free energy from the PB calculation, including both:

- The **Coulombic energy** of the charges in the dielectric medium
- The **solvation (reaction field) energy**

This tells you the net electrostatic cost or benefit of bringing protein and ligand together. It includes the **desolvation penalty** — when the ligand binds, it loses favorable interactions with water, and the protein–ligand electrostatic interactions have to compensate for this loss.

### Application to catalysis

For catalysis specifically, what you want is:

$$\Delta\Delta G_{\text{elec}} = \Delta G_{\text{elec}}^{\text{TS}} - \Delta G_{\text{elec}}^{\text{GS}}$$

If this is **negative**, the enzyme preferentially stabilizes the transition state electrostatically — the definition of electrostatic catalysis. Computing this requires a model of the TS charge distribution (from QM calculations or literature TS models), but the PB machinery is identical.

---

## 9. Key Assumptions and Their Validity

### 9.1 Continuum dielectric model

Replaces the granular reality of discrete water molecules and protein atoms with smooth dielectric regions. Works well for net electrostatic effects but **misses specific water-mediated hydrogen bonds**. For catalysis, specific water molecules in the active site can matter enormously, and PB won't capture those.

### 9.2 Choice of protein dielectric ($\epsilon = 2$–$4$)

This is the single most debated parameter:

- $\epsilon = 2$ represents only electronic polarization.
- $\epsilon = 4$ implicitly includes some nuclear relaxation (small backbone and sidechain rearrangements).

**For comparing variants of the same enzyme, the choice matters less** because the errors are systematic. For absolute binding energies, it matters a great deal.

### 9.3 Static structure assumption

You're computing the potential for one conformation. Real enzymes breathe and fluctuate. If your active site undergoes significant conformational change upon substrate binding, a single-structure PB calculation may miss important features. MD-averaged PB (solving PB on multiple frames from a molecular dynamics trajectory) helps, at the cost of much more compute.

### 9.4 Linearization

Usually fine for proteins at physiological conditions, but can underestimate the potential very close to clusters of like charges (e.g., a poly-Asp patch). For active site calculations, this rarely matters.

---

## 10. Connecting Back to Enzyme Engineering

When you mutate a residue in or near the active site, you're changing:

- $\rho(\mathbf{r})$ — the charge distribution
- Potentially $\epsilon(\mathbf{r})$ — if you change the local packing and hydration

The PB-EC metric lets you ask: **did this mutation make the electrostatic environment more or less complementary to what the transition state needs?**

A variant library scored by EC, combined with other metrics (shape complementarity, catalytic geometry, stability proxies), gives you a physically grounded ranking that goes well beyond sequence-level features. The key advantage of PB-EC over simpler metrics is that it correctly accounts for the dielectric environment, which is the physical origin of enzymatic electrostatic catalysis in the first place.

---

## 11. Implementation: `artios/electrostatics/`

This section describes the algorithm as implemented in the codebase. The code lives in `artios/electrostatics/` with five modules:

```
artios/electrostatics/
├── __init__.py     # Public API: compute_ec, ECResult, ECConfig, split_complex, ...
├── core.py         # Pipeline orchestration: compute_ec(), split_complex()
├── surface.py      # Shrake-Rupley interface surface point generation
├── apbs.py         # APBS wrapper: grid sizing, input generation, DX parsing
├── pqr.py          # PQR generation (PDB2PQR for protein, MMFF94 for ligand)
└── elements.py     # PARSE force field VdW radii
```

### 11.1 Pipeline Overview

```
                    ┌─────────────┐
                    │  Input CIF  │
                    │  (complex)  │
                    └──────┬──────┘
                           │
                    split by chain ID
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
          ┌──────────┐         ┌──────────┐
          │ Protein  │         │  Ligand  │
          └────┬─────┘         └────┬─────┘
               │                    │
    ┌──────────┤                    │
    │          │                    │
    ▼          ▼                    ▼
 Surface    PDB2PQR             MMFF94
 points     (PARSE)            charges
    │          │                    │
    │          ▼                    ▼
    │      protein.pqr         ligand.pqr
    │          │                    │
    │          ▼                    ▼
    │      APBS (LPB)          APBS (LPB)
    │          │                    │
    │          ▼                    ▼
    │      φ_protein(r)        φ_ligand(r)
    │          │                    │
    └──────────┤────────────────────┘
               │
        Sample both φ at
        surface points
               │
               ▼
      Pearson correlation
               │
               ▼
           EC score
```

`compute_ec(structure, ligand_chain_id)` runs this entire pipeline and returns an `ECResult` with the EC score, number of interface points, and the raw potential arrays.

### 11.2 Surface Point Generation (Shrake-Rupley)

**Module:** `surface.py`

Rather than sampling potentials at atom positions (which gives nonuniform, sparse coverage), we generate points on the protein's **solvent-accessible surface** (SAS) at the protein-ligand contact zone. This follows the standard Shrake-Rupley approach used in the EC literature.

**Algorithm:**

1. **Fibonacci sphere template.** Generate $n$ evenly-spaced points on a unit sphere using the golden spiral (Fibonacci lattice):

$$\theta_i = \frac{2\pi i}{\varphi}, \quad \phi_i = \arccos\left(1 - \frac{2(i + 0.5)}{n}\right)$$

where $\varphi = (1 + \sqrt{5})/2$ is the golden ratio. This gives near-optimal point distributions without iterative optimization. Default: $n = 100$ points per atom.

2. **Pre-filter protein atoms.** Using a KDTree on ligand coordinates, identify protein atoms close enough that their SAS spheres could contribute contact points. The cutoff is: $d_{\text{atom-to-ligand}} \leq r_{\text{SAS,max}} + r_{\text{VdW,ligand,max}} + \text{contact\_cutoff}$.

3. **For each nearby protein atom:**
   - Place $n$ Fibonacci points on its SAS sphere (center + $(r_{\text{VdW}} + r_{\text{probe}}) \times \hat{u}_i$).
   - **Burial check:** Remove points that fall inside any other protein atom's SAS sphere. This is the core of the Shrake-Rupley algorithm — only solvent-exposed points survive. Uses `query_ball_point` on a protein KDTree for efficiency.
   - **Contact filter:** Of the surviving (exposed) points, keep only those within `contact_cutoff` (default 1.4 Å, one probe radius) of the ligand's VdW surface. Distance to the ligand surface is computed as: $d_{\text{surface}} = d_{\text{to nearest ligand atom}} - r_{\text{VdW, that atom}}$.

4. **Concatenate** all surviving points → interface surface points (typically 200–400 points).

The contact filter gives a tight shell of points right at the protein-ligand contact, not a generous envelope. VdW radii come from the PARSE force field (`elements.py`).

### 11.3 Grid Focusing

**Module:** `apbs.py` → `focused_grid_dims()`

APBS uses the `mg-auto` sequential focusing method: a coarse grid is solved first to provide boundary conditions for a fine grid. Both grids share the same center point (the binding site centroid).

**Fine grid** — covers only the binding site at 0.5 Å resolution:
- Extent = ptp(binding site coords) + 2 × padding (20 Å)
- `dime` = ceil(extent / 0.5), made odd (APBS requirement)
- `fglen` = dime × 0.5

**Coarse grid** — covers the full protein + ligand for correct long-range boundary conditions:
- Half-extent = max(|all_coords − center|) + padding along each axis
- `cglen` = 2 × half-extent, with the constraint `cglen` ≥ 2 × `fglen` (APBS stability requirement)

This gives ~7× fewer grid points compared to a uniform fine grid covering the entire protein (e.g., 1.5M vs 11.2M for a typical 300-residue enzyme).

### 11.4 Charge and Radius Assignment

**Module:** `pqr.py`

**Protein** — PDB2PQR with the PARSE force field:
- The protein AtomArray is written to a temporary PDB file
- `pdb2pqr30 --ff=PARSE --noopt --nodebump --keep-chain` assigns partial charges and radii
- Multi-character mmCIF chain IDs are remapped to single characters for PDB format compatibility
- Output: a `.pqr` file (PDB format + charge + radius per atom)

**Ligand (CCD codes)** — MMFF94 charges via RDKit:
- Load the pre-built RDKit mol from the CCD component cache
- Add hydrogens (`Chem.AddHs`) for correct MMFF94 atom typing
- Extract partial charges per heavy atom via `MMFFGetMoleculeProperties`
- Map charges back to the AtomArray ordering using `mol.atom_map`
- Radii are PARSE VdW radii looked up by element

**Ligand (SMILES, for designs)** — same MMFF94 approach, but with an additional step:
- Generate a 3D conformer from SMILES (`ETKDGv3`)
- Hungarian matching maps SMILES heavy atoms to CIF ligand atoms by element + spatial distance
- Charges transfer through this mapping

### 11.5 Poisson-Boltzmann Solve

**Module:** `apbs.py` → `generate_apbs_input()`, `run_apbs()`

APBS is invoked twice — once for the protein alone, once for the ligand alone — with identical grid parameters (center, dime, fglen, cglen). Key APBS settings:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `lpbe` | — | Linearized Poisson-Boltzmann equation |
| `pdie` | 2.0 | Protein interior dielectric |
| `sdie` | 78.54 | Solvent dielectric (water at 25°C) |
| `ion` | ±1, 0.15 M | Monovalent salt at physiological ionic strength |
| `srfm smol` | — | Molecular surface for dielectric boundary |
| `chgm spl2` | — | Cubic B-spline charge mapping to grid |
| `srad` | 1.4 Å | Solvent probe radius |
| `bcfl sdh` | — | Single Debye-Hückel boundary conditions on coarse grid |
| `temp` | 298.15 K | Temperature |

Output: a `.dx` file containing $\phi(\mathbf{r})$ at every grid point in $k_BT/e$ units.

### 11.6 Sampling and Correlation

**Module:** `apbs.py` → `sample_potential_at_points()`

Both potential grids ($\phi_{\text{protein}}$, $\phi_{\text{ligand}}$) are sampled at the interface surface points via **trilinear interpolation** (`scipy.ndimage.map_coordinates` with `order=1`). World coordinates are converted to fractional grid indices: $\text{idx} = (\mathbf{r} - \text{origin}) / \text{delta}$.

The EC score is the Pearson correlation of the two potential vectors:

$$EC = \text{corr}(\phi_{\text{protein}}, \phi_{\text{ligand}})$$

Negative EC = the protein creates negative potential where the ligand is positive (and vice versa) = electrostatic complementarity. The result object also exposes the raw potential arrays for per-point analysis.

### 11.7 Default Parameters Summary

| Parameter | Default | Set in |
|-----------|---------|--------|
| Fine grid spacing | 0.5 Å | `ECConfig.fine_grid_spacing` |
| Grid padding | 20.0 Å | `ECConfig.grid_padding` |
| Protein dielectric | 2.0 | `ECConfig.protein_dielectric` |
| Solvent dielectric | 78.54 | `ECConfig.solvent_dielectric` |
| Ionic strength | 0.15 M | `ECConfig.ionic_strength` |
| Probe radius | 1.4 Å | `ECConfig.probe_radius` |
| Surface points/atom | 100 | `interface_surface_points` |
| Contact cutoff | 1.4 Å | `interface_surface_points` |
| Interface cutoff (grid sizing) | 5.0 Å | `core.INTERFACE_CUTOFF` |

---

## References and Further Reading

- Warshel, A. (1998). Electrostatic origin of the catalytic power of enzymes and the role of preorganized active sites. *J. Biol. Chem.*, 273(42), 27035–27038.
- Baker, N. A., et al. (2001). Electrostatics of nanosystems: application to microtubules and the ribosome. *Proc. Natl. Acad. Sci.*, 98(18), 10037–10041. (APBS)
- Fried, S. D., & Boxer, S. G. (2017). Electric fields and enzyme catalysis. *Annu. Rev. Biochem.*, 86, 387–415.
- McCoy, A. J., Chandana Epa, V., & Colman, P. M. (1997). Electrostatic complementarity at protein/protein interfaces. *J. Mol. Biol.*, 268(2), 570–584.
- Honig, B., & Nicholls, A. (1995). Classical electrostatics in biology and chemistry. *Science*, 268(5214), 1144–1149.
