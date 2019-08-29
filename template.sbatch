####  SBATCH preamble

#SBATCH --mail-user=mduschen@umich.edu
#SBATCH -m ae

#SBATCH -A e
#SBATCH -l qos=flux
#SBATCH -q flux

#SBATCH -l nodes=1:ppn=2,pmem=2gb
#SBATCH -l walltime=1:15:00
#SBATCH -j oe
#SBATCH -V
