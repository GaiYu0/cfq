FLAGFILE=config/mcd1/2021.02.02_mcd1_final.1.cfg sbatch --nodelist=pavia --gres="gpu:1" scripts/run.sh
FLAGFILE=config/mcd2/2021.02.02_mcd2_final.1.cfg sbatch --nodelist=luigi --gres="gpu:1" scripts/run.sh
FLAGFILE=config/mcd3/2021.02.02_mcd3_final.1.cfg sbatch --nodelist=zanino --gres="gpu:1" scripts/run.sh
notify.py "Launched set 1"
sleep 600

FLAGFILE=config/mcd1/2021.02.02_mcd1_final.2.cfg sbatch --nodelist=pavia --gres="gpu:1" scripts/run.sh
FLAGFILE=config/mcd2/2021.02.02_mcd2_final.2.cfg sbatch --nodelist=luigi --gres="gpu:1" scripts/run.sh
FLAGFILE=config/mcd3/2021.02.02_mcd3_final.2.cfg sbatch --nodelist=zanino --gres="gpu:1" scripts/run.sh
notify.py "Launched set 2"
sleep 600

FLAGFILE=config/mcd1/2021.02.02_mcd1_final.3.cfg sbatch --nodelist=pavia --gres="gpu:1" scripts/run.sh
FLAGFILE=config/mcd2/2021.02.02_mcd2_final.3.cfg sbatch --nodelist=luigi --gres="gpu:1" scripts/run.sh
FLAGFILE=config/mcd3/2021.02.02_mcd3_final.3.cfg sbatch --nodelist=zanino --gres="gpu:1" scripts/run.sh
notify.py "Launched set 3"
sleep 600

FLAGFILE=config/mcd1/2021.02.02_mcd1_final.4.cfg sbatch --nodelist=pavia --gres="gpu:1" scripts/run.sh
FLAGFILE=config/mcd2/2021.02.02_mcd2_final.4.cfg sbatch --nodelist=luigi --gres="gpu:1" scripts/run.sh
FLAGFILE=config/mcd3/2021.02.02_mcd3_final.4.cfg sbatch --nodelist=zanino --gres="gpu:1" scripts/run.sh
notify.py "Launched set 4"
sleep 600

FLAGFILE=config/mcd1/2021.02.02_mcd1_final.5.cfg sbatch --nodelist=pavia --gres="gpu:1" scripts/run.sh
FLAGFILE=config/mcd2/2021.02.02_mcd2_final.5.cfg sbatch --nodelist=luigi --gres="gpu:1" scripts/run.sh
FLAGFILE=config/mcd3/2021.02.02_mcd3_final.5.cfg sbatch --nodelist=zanino --gres="gpu:1" scripts/run.sh
notify.py "Launched set 5"
sleep 600

FLAGFILE=config/mcd1/2021.02.02_mcd1_final.6.cfg sbatch --nodelist=pavia --gres="gpu:1" scripts/run.sh
FLAGFILE=config/mcd2/2021.02.02_mcd2_final.6.cfg sbatch --nodelist=luigi --gres="gpu:1" scripts/run.sh
FLAGFILE=config/mcd3/2021.02.02_mcd3_final.6.cfg sbatch --nodelist=zanino --gres="gpu:1" scripts/run.sh
notify.py "Launched set 6"