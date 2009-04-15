
export OMP_NUM_THREADS=1
python test_ode_omp.py
export OMP_NUM_THREADS=2
python test_ode_omp.py
export OMP_NUM_THREADS=3
python test_ode_omp.py
export OMP_NUM_THREADS=4
python test_ode_omp.py
