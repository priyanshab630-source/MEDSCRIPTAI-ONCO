# PyMOL session for complex.pdb
#   pymol -cq view.pml          (headless, writes docking.png)
load /scratch/work/medscriptai-onco/runs/4ajl/complex.pdb, complex
hide everything
show cartoon, polymer
color grey70, polymer
select ligand, resn UNL
show sticks, ligand
color yellow, ligand
select pocket, byres (polymer within 4.5 of ligand)
show sticks, pocket and not name N+C+O
color cyan, pocket and elem C
distance hbonds, ligand, polymer, 3.5, mode=2
show dashes, hbonds
color red, hbonds
label pocket and name CA, "%s%s" % (resn, resi)
set label_size, 14
orient ligand
bg_color white
ray 1600, 1200
png docking.png, dpi=200
