# Create the Apptainer image
cd apptainer || exit
apptainer build offnav.sif offnav.def
cd ..

# Init apptainer with home mounted
apptainer run --nv \
  --bind /gs/fs/tga-aklab/data \
  --bind /gs/fs/tga-aklab/carlos/repositorios \
  --bind /gs/fs/tga-aklab/carlos/miniconda3 \
  apptainer/offnav.sif

# Run inside apptainer (I don't know if this will work)
bash scripts/install_conda.sh