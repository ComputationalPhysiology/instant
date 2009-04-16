
export OMP_NUM_THREADS=1
python _test_ode_omp.py
export OMP_NUM_THREADS=2
python _test_ode_omp.py
export OMP_NUM_THREADS=3
python _test_ode_omp.py
export OMP_NUM_THREADS=4
python _test_ode_omp.py
