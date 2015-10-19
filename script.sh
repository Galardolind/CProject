rm var
echo "STARTTTTT"
echo ""
export OMP_NUM_THREADS=4
for (( i=0; i < 25; i=i + 1));
do 
	echo "test numero :  $i " >> var 
	echo "" >> var 
	mpirun -np $i -x OMP_NUM_THREADS $1 test >> var
	echo "" >> var
	echo "/////////////////////////////////////////////////////////////////" >> var 
	echo "" >> var
done
echo ""
echo "Fin test classique"
echo "Fin test classique" >> var
echo ""
mpirun -np 3 -x OMP_NUM_THREADS $1 testdigit >> var
echo ""
echo "Fin test digit (nombre à 2 chiffre)"
echo "Fin test digit (nombre à 2 chiffre)" >> var
echo "" >> var
echo "test 256*256 matrice 1 proco : " >> var 
echo "" >> var
mpirun -np 1 -x OMP_NUM_THREADS $1 test256 >> var
echo "" >> var
echo "/////////////////////////////////////////////////////////////////" >> var 
echo ""
echo "Fin test 1 proco 256"
echo "Fin test 1 proco 256" >> var
echo "" >> var
echo "test 256*256 matrice 8 proco : " >> var 
echo "test 256*256 matrice 8 proco : "
echo ""
echo "" >> var
mpirun -np 1 -x OMP_NUM_THREADS $1 test256 >> var
echo ""
echo "Fin test 8 proco 256 "
echo "Fin test 8 proco 256 " >> var
echo "" >> var
echo "/////////////////////////////////////////////////////////////////" >> var 
echo ""
echo "" >> var
echo "test 1024*1024 matrice 1 proco : " >> var 
echo "test 1024*1024 matrice 1 proco : "
echo ""
echo "" >> var
mpirun -np 1 -x OMP_NUM_THREADS $1 test1024 >> var
echo ""
echo "Fin test 1 proco 1024 "
echo "Fin test 1 proco 1024 " >> var
echo "/////////////////////////////////////////////////////////////////" >> var 
echo ""
echo "" >> var
echo "test 1024*1024 matrice 8 proco : " >> var 
echo "" >> var
mpirun -np 8 -x OMP_NUM_THREADS $1 test1024 >> var
echo ""
echo " test 8 proco 2048 "
echo " test 8 proco 2048 " >> var
mpirun -np 8 -x OMP_NUM_THREADS $1 test2048 >> var
