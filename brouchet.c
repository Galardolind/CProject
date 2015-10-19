#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <omp.h>



void doMatrixMatrix(int rank, int numprocs, int M, int N, int P, MPI_Comm comm, char* filename);
void doMatricialScatter(int * send, int * rcv, int * nbLinePerProc, int col, int numprocs, int rank, MPI_Comm comm);
void doMatricialGather(int * matrice, int * rcv, int * nbLinePerProc, int col, int numprocs, int rank, MPI_Comm comm);
void doCirculation(int * A, int * B, int * C, int N, int P, int numprocs, int rank, int * nbLineAPerProc, int * nbLineBPerProc);

/**
 * Lit le fichier pour compter le nombre de données (ici le nombre de sommets).
 * @param fileName Nom du fichier
 * @return Un entier représentant le nombre de données (ici le nombre de sommets)
 */
int getDataSize(char* filename) {
    FILE *file;
    int nb = 0;
    int k;
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Impossible d'ouvrir le fichier\n");
        exit(EXIT_FAILURE);
    }
    while (fscanf(file, "%d", &k) != EOF) {
        nb++;
    }
    fclose(file);
    return nb;
}

/**
 * Lit la première ligne du fichier pour en déduire le nombre de donnée par lignes
 * @param filename
 * @return Un entier représentant le nombre de données par ligne 
 */
int sizeLine(char* filename) {
    FILE *file;
    int nb = 1;
    int c;
    int lastC = 1;
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Impossible d'ouvrir le fichier\n");
        exit(EXIT_FAILURE);
    }
    while ((c = fgetc(file)) != EOF) {
        if ('\n' == c || c == 13) {
            if (lastC == 1) {
                nb--;
            }
            break;
        }
        if (c == ' ') {
            if (lastC == 0) {
                nb++;
            }
            lastC = 1;
        } else {
            lastC = 0;
        }
    }
    fclose(file);
    return nb;
}

/**
 * Lit le fichier et met les entier dans le tableau passé en paramètre.
 * @param filename nom du fichier
 * @param A tableau d'entier déjà alloué.
 */
void loadData(char* filename, int * A) {
    FILE *file;
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Impossible d'ouvrir le fichier\n");
        exit(EXIT_FAILURE);
    }
    int i = 0;
    int k;
    while (fscanf(file, "%d", &k) != EOF) {
        A[i] = k;
        i++;
    }
    fclose(file);
}

/**
 * Remplis le taleau B en suivant l'algorithme de Floyd-Warshall
 * @param A tableau représentant la matrice de base.
 * @param B tableau resultant de la matrice de base (W dans l'ennoncé)
 * @param M nombre de ligne de A
 * @param N nombre de colonne de A
 */
void fillBAdj(int * A, int * B, int M, int N) {
    int i;
    if (M > N) {
        #pragma omp parallel for
        for (i = 0; i < N; i++) {
            int j;
            for (j = 0; j < N; j++) {
                if (i == j) {
                    B[i * N + j] = 0;
                } else if (A[i * N + j] != 0) {
                    B[i * N + j] = A[i * N + j];
                }
            }
        }
    } else {
        #pragma omp parallel for
        for (i = 0; i < M; i++) {
            int j;
            for (j = 0; j < N; j++) {
                if (i == j) {
                    B[i * N + j] = 0;
                } else if (A[i * N + j] != 0) {
                    B[i * N + j] = A[i * N + j];
                }
            }
        }

    }

    if (M < N) {
        #pragma omp parallel for
        for (i = M; i < N; i++) {
            int j;
            for (j = 0; j < N; j++) {
                if (i == j) {
                    B[i * N + j] = 0;
                }
            }
        }
    }
}

/**
 * Remplit la matrice d'entier aleatoire entre 0 et 10.
 * @param m tableau représentant la matrice
 * @param nbLine nombre de ligne de m
 * @param nbCol nombre de colonne de m
 */
void fillMatrice(int* m, int nbLine, int nbCol) {
    int i;
    for (i = 0; i < nbLine * nbCol; i++) {
        m[i] = rand() % 10;
    }
}

/**
 * Affiche un tableau d'entier.
 * @param tab tableau a afficher
 * @param size nombre de donnée dans le tableau
 */
void printTab(int * tab, int size) {
    int i;
    printf("[");
    for (i = 0; i < size; i++) {
        printf(" %d", tab[i]);
    }
    printf(" ]\n");
}

/**
 * Affiche une matrice sous forme de carré :) 
 * @param m matrice a afficher sous forme de tableau
 * @param nbLigne nombre de ligne dans m
 * @param nbCol nombre de colonne dans m
 * @param myrank rank du processeur
 */
void printMatrice(int* m, int nbLigne, int nbCol, int myrank) {
    int i;
    printf("\n\t===Proco : %d==== ", myrank);
    printf("\n\t  ");
    for (i = 0; i < nbCol; i++) {
        printf("___");
    }
    for (i = 0; i < nbLigne * nbCol; i++) {
        if (i % nbCol == 0) {
            if (i != 0) {
                printf(" |");
            }
            printf("\n\t| ");
        }
        printf("%2d ", m[i]);
    }
    printf(" |");
    printf("\n\t  ");
    for (i = 0; i < nbCol; i++) {
        printf("___");
    }
    printf("\n\n");
}

/**
 * Affichage demandé.
 * @param m matrice sous forme de tableau
 * @param nbLigne nombre de ligne de m
 * @param nbCol nombre de colonne de m
 */
void huetPrint(int* m, int nbLigne, int nbCol) {
    int i;
    int first = 1;
    for (i = 0; i < nbLigne * nbCol; i++) {
        if (i % nbCol == 0 && first != 1) {
            printf("\n");
        }
        if (m[i] == INT_MAX) {
            printf("-1 ");
        } else {
            printf("%d ", m[i]);
        }
        first = 0;
    }
    printf("\n");
}

/**
 * opération d'ajout de 3 élements
 * @param C valeur a ajouté 1 
 * @param A valeur a ajouté 2
 * @param B valeur a ajouté 3
 * @return un entier étant la somme des trois éléments
 */
int addElement(int C, int A, int B) {
    return C + A + B;
}

/**
 * opération de multiplication de 3 élements
 * @param C valeur a multiplier 1 
 * @param A valeur a multiplier 2
 * @param B valeur a multiplier 3
 * @return un entier étant la multiplication des trois éléments
 */
int multElement(int C, int A, int B) {
    return C + (A * B);
}

/**
 * Algorithme de Floyd-Warshall
 * @param C case de la matrice initiale
 * @param A case de la matrice A
 * @param B case de la matrice B
 * @return le min entre C et A+B
 */
int shortestPath(int C, int A, int B) {
    int result;
    if (A == INT_MAX || B == INT_MAX) {
        result = INT_MAX;
    } else {
        result = A + B;
    }
    if (result > C) {
        result = C;
    }
    return result;
}

/**
 * Le calcul qui a tuer des millions de codeur à la recherche des indices magique !
 * @param A matrice linéarisée, second membre de l'opération opp
 * @param B matrice linéarisée, troisième membre de l'opération opp
 * @param C matrice ou le résultat va être inscrit, premier membre de l'opération opp
 * @param M nombre de ligne de A
 * @param NA nombre de colonne de A
 * @param NB nombre de ligne de B
 * @param P nombre de colonne de B
 * @param tour decalage du au placement du processeur dans la liste et du tour de la circulation
 * @param opp oppération a effectuer
 * @param rank rank du processeur (utile uniquement en debug)
 */
void calcul(int * A, int * B, int * C, int M, int NA, int NB, int P, int tour, int (*opp) (int, int, int), int rank) {
    int i;
    /*
        if (rank == 0) {
            printf("C : ");
            printMatrice(C, M, P, rank);
            printf("A : ");
            printMatrice(A, M, NA, rank);
            printf("B : ");
            printMatrice(B, NB, P, rank);
            printf("\nM : %d, NA : %d, NB : %d,P : %d, tour : %d", M, NA, NB, P, tour);
        }
     */
	 
	#pragma omp parallel for
    for (i = 0; i < M; i++) {
        int j;
        for (j = 0; j < P; j++) {
            /*
                        if (rank == 0) {
                            //printf("\n\nC[%d] <> ", i * P + j);
                            printf("\n\nC[%d] <> ", C[i * P + j]);
                        }
             */
            int k;
            for (k = 0; k < NB; k++) {
                /*
                                if (rank == 0) {
                                    //printf("A[%d] + B[%d] <> ", i * NA + k + ((tour) % NA), k * P + j);
                                    printf("A[%d] + B[%d] <> ", A[i * NA + k + (tour % NA)], B[k * P + j]);
                                }
                 */
                C[i * P + j] = (*opp)(C[i * P + j], A[i * NA + k + (tour % NA)], B[k * P + j]);
            }

        }
    }
    /*
        if (rank == 0) {
            printf("\n\n");
            printf("C : ");
            printMatrice(C, M, P, rank);
        }
     */
}

/**
 * Calcul le nombre de tour pour arrivé à W^n sachant que si ce n'est pas un multiple
 * de 2 il va donc faire la puissance de 2 inférieur à n et compléter avec les tours.
 * @param N nombre d'élément
 * @return Le nombre de tour minimum pour arrivé à N.
 */
int calculNbTour(int N){
    if(N < 2){
        return 1;
    }
    int x = 2;
    int m = 0;
    while(x <= N){
        m++;
        x *= 2;
    }
    x /= 2;
    if(x != N){
        m = m + (N-x);
    }
    return m;
}

/**
 * Calcul du nombre de ligne de matrice necessaire par processeur pour que cela soit
 * au maximum equitable entre chaques.
 * 
 * @param nbLine nombre de ligne de la matrice
 * @param nbproc nombre de processeur
 * @return un tableau contenant la liste du nombre de ligne/processeur 
 */
int * nbPerProc(int nbLine, int nbproc) {
    int packSize = (int) (nbLine / nbproc);
    int * nbLinePerProc = malloc(sizeof (int)*nbproc);
    int i;
    for (i = 0; i < nbproc; i++) {
        nbLinePerProc[i] = packSize;
    }
    if (packSize == 0) {
        #pragma omp parallel for
        for (i = 0; i < nbLine; i++) {
            nbLinePerProc[i] = nbLinePerProc[i] + 1;
        }
    } else if (nbLine % (nbproc * packSize) != 0) {
        #pragma omp parallel for
        for (i = 0; i < nbLine % (nbproc * packSize); i++) {
            nbLinePerProc[i] = nbLinePerProc[i] + 1;
        }
    }
    return nbLinePerProc;
}

/**
 * Ancienne version d'allocation de matrice a deux dimension.
 * @param nbLin nombre de ligne
 * @param nbCol nombre de colonne
 * @return un tableau à deux dimension alloué correctement
 */
int **createMatrice(int nbLin, int nbCol) {
    int **tableau = (int **) malloc(sizeof (int*)*nbLin);
    int *tableau2;
    int i;
    for (i = 0; i < nbLin; i++) {
        tableau2 = (int *) malloc(sizeof (int)*nbCol);
        tableau[i] = tableau2;
    }
    return tableau;
}

/**
 * Alloue une matrice de façon linéarisé.
 * @param nbLin nombre de ligne
 * @param nbCol nombre de colonne
 * @return  un tableau contenant le nombre de case de la matrice.
 */
int* createMatriceLinearise(int nbLin, int nbCol) {
    int* linearMatrice = (int *) malloc(nbLin * sizeof (int)*nbCol);
    if (linearMatrice == NULL) {
        printf("Impossible d'allouer une matrice ! Pc en carton ! \n");
        exit(EXIT_FAILURE);
    }
    return linearMatrice;
}

/**
 * Le main contient le calcul de la taille de la matrice et la gestion du nombre
 * de processeur (si N < numprocs) alors le nombre de processeur sera diminué
 * pour qu'il y ait au minimum 1 ligne par processeur dans le groupe.
 * @param argc
 * @param argv
 * @return 
 */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Stop jouer au coquin avec vos %d arguments !\n", argc);
        exit(EXIT_FAILURE);
    }

    /*
     * La solution la plus rapide pour savoir la taille d'une matrice N * N 
     * dans un fichier serait de compter le nombre d'entier sur une ligne.
     * Mais ça ne marchera pas pour les matrices M * N.
     */

    char * filename = argv[1];
    int sizeL = sizeLine(filename);
    int nbData = getDataSize(filename);

    int M = nbData / sizeL;
    int N = sizeL;
    int P = sizeL;

    if (M <= 0 || N <= 0 || P <= 0) {
        printf("Stop jouer au coquin en mettant n'importe quoi ! M: %d, N: %d, P: %d, si vous avez mis une ligne unique rajoutez un retour à la ligne\n",M,N,P);
        exit(EXIT_FAILURE);
    }

    int rank, numprocs, grp_rank, grp_size;
    MPI_Comm new_comm;
    MPI_Group orig_group;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Group_rank(orig_group, &grp_rank);
    MPI_Group_size(orig_group, &grp_size);
    MPI_Comm_create(MPI_COMM_WORLD, orig_group, &new_comm);

    /*
     * Exclusion des processeur inutile.
     * On empeche les étape de circulations inutile (avec des processeurs vide)
     * en diminuant le nombre de processeur de façon a avoir au minimum 1 ligne
     * par processeur.
     */
    if (M < numprocs || N < numprocs) {
        int size;
        int start;
        if (N < M) {
            size = numprocs - N;
            start = N;
        } else {
            size = numprocs - M;
            start = M;
        }
        int range[size];
        int i, j;
        for (i = start, j = 0; i < numprocs; i++, j++) {
            range[j] = i;
        }
        MPI_Group_excl(orig_group, size, range, &orig_group);
        MPI_Group_rank(orig_group, &grp_rank);
        MPI_Group_size(orig_group, &grp_size);
        MPI_Comm_create(MPI_COMM_WORLD, orig_group, &new_comm);

    }

    if (rank == grp_rank) {
        doMatrixMatrix(rank, grp_size, M, N, P, new_comm, filename);
    }

    MPI_Finalize();
    exit(EXIT_SUCCESS);
}

/**
 * Cette methode éxecute l'ensemble des fonctions nécessaire a l'obtention du
 * résultat.
 * Elle s'occupe aussi d'allouer les tableaux de chaque processeur.
 * @param rank rank du processeur
 * @param numprocs nombre de processeur
 * @param M nombre de ligne de A
 * @param N nombre de colonne de A (ou ligne de B)
 * @param P nombre de colonne de B
 * @param comm communication de MPI
 * @param filename nom du fichier a load
 * @return 
 */
void doMatrixMatrix(int rank, int numprocs, int M, int N, int P, MPI_Comm comm, char* filename) {

    int * nbLineAPerProc = nbPerProc(N, numprocs);
    int * nbLineBPerProc = nbPerProc(N, numprocs);

    int * A;
    int * B;
    int * C;
    int * R;

    if (rank == 0) {
        A = createMatriceLinearise(M, N);
        B = createMatriceLinearise(N, P);
        C = createMatriceLinearise(N, P);
        R = createMatriceLinearise(N, P);
        int x;
        for (x = 0; x < N * P; x++) {
            C[x] = INT_MAX;
            B[x] = INT_MAX;
        }
    } else {
        // ils auront tous a un moment T au minimum le nombre de lignes du proco 0
        A = createMatriceLinearise(nbLineAPerProc[0], P);
        B = createMatriceLinearise(nbLineBPerProc[0], P);
        C = createMatriceLinearise(nbLineAPerProc[0], P);
        int x;
        for (x = 0; x < nbLineAPerProc[rank] * P; x++) {
            C[x] = INT_MAX;
            B[x] = INT_MAX;
            A[x] = INT_MAX;
        }
    }

    if (rank == 0) {
        loadData(filename, A);
        fillBAdj(A, B, M, N);
        /*
                printf("\t\tA\n");
                printMatrice(A, M, N, rank);
                printf("\t\tB\n");
                printMatrice(B, N, N, rank);
         */
        int * tmp = (int*) realloc(A, N * P * sizeof (int));
        if (tmp != NULL) {
            A = tmp;
        } else {
            MPI_Finalize();
            printf("An error during realloc operation ! \n");
            exit(EXIT_FAILURE);
        }
    }

    int nbTour = calculNbTour(N);
    int i;
    // on fait n fois pour faire W^n
    for (i = 0; i < nbTour; i++) {
        // si il n'y a qu'un processeur pas besoin de faire autre chose que le calcul
        if (numprocs > 1) {
            doMatricialScatter(B, B, nbLineBPerProc, P, numprocs, rank, comm);
            doMatricialScatter(B, A, nbLineAPerProc, P, numprocs, rank, comm);
            doCirculation(A, B, C, N, P, numprocs, rank, nbLineBPerProc, nbLineBPerProc);
            doMatricialGather(C, R, nbLineBPerProc, P, numprocs, rank, comm);
            if (rank == 0) {
                int x;
                for (x = 0; x < N * P; x++) {
                    B[x] = R[x];
                    C[x] = INT_MAX;
                }
            } else {
                int x;
                for (x = 0; x < nbLineAPerProc[rank] * P; x++) {
                    C[x] = INT_MAX;
                    B[x] = INT_MAX;
                    A[x] = INT_MAX;
                }
            }

        } else {
            calcul(B, B, C, N, N, N, P, 0, shortestPath, rank);
            int x;
            for (x = 0; x < N * P; x++) {
                B[x] = C[x];
            }
        }
    }
    if (rank == 0) {
        if (numprocs > 1) {
            huetPrint(R, N, P);
        } else {
            huetPrint(C, N, P);
        }
    }
}

/**
 * Execute la fonction scatterv de MPI pour une matrice donnée en paramètre.
 * @param send matrice a envoyer
 * @param rcv matrice pré alloué ou recevoir
 * @param nbLinePerProc tableau du nombre de ligne pour chaque processeur la case 0 correspondant au rank 0
 * @param col nombre de colonne dans la matrice
 * @param numprocs nombre de processeur
 * @param rank rank du processeur
 * @param comm Communication de MPI pour le groupe courant
 * @return 
 */
void doMatricialScatter(int * send, int * rcv, int * nbLinePerProc, int col, int numprocs, int rank, MPI_Comm comm) {
    int i, *display, *scounts;

    if (rank == 0) {
        display = (int *) malloc(numprocs * sizeof (int));
        scounts = (int *) malloc(numprocs * sizeof (int));
        if (scounts == NULL || display == NULL) {
            printf("Problème lors de l'allocation du tableau de %d entier \n", numprocs);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
        scounts[0] = nbLinePerProc[0] * col;
        display[0] = 0;
        for (i = 1; i < numprocs; i++) {
            scounts[i] = nbLinePerProc[i] * col;
            display[i] = display[i - 1] + scounts[i - 1];
        }
    }
    MPI_Scatterv(send, scounts, display, MPI_INT, rcv, nbLinePerProc[rank] * col, MPI_INT, 0, comm);

    if (rank == 0) {
        free(display);
        free(scounts);
    }
}

/**
 * Execute la fonction de Gatherv sur les processeurs.
 * @param matrice Matrice ou est stocké la matrice a réunir dans chaque processeur
 * @param rcv Matrice ou receptionner tous les bouts de matrice
 * @param nbLinePerProc tableau du nombre de ligne pour chaque processeur la case 0 correspondant au rank 0
 * @param col nombre de colonne de la matrice
 * @param numprocs nombre de processeurs
 * @param rank rank du processeur
 * @param comm Communication de MPI pour le groupe courant
 * @return 
 */
void doMatricialGather(int * matrice, int * rcv, int * nbLinePerProc, int col, int numprocs, int rank, MPI_Comm comm) {
    int i, *display, *scounts;
    if (rank == 0) {
        display = (int *) malloc(numprocs * sizeof (int));
        scounts = (int *) malloc(numprocs * sizeof (int));
        if (scounts == NULL || display == NULL) {
            printf("Problème lors de l'allocation du tableau de %d entier \n", numprocs);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
        scounts[0] = nbLinePerProc[0] * col;
        display[0] = 0;
        for (i = 1; i < numprocs; i++) {
            scounts[i] = nbLinePerProc[i] * col;
            display[i] = display[i - 1] + scounts[i - 1];
        }
    }
    MPI_Gatherv(matrice, nbLinePerProc[rank] * col, MPI_INT, rcv, scounts, display, MPI_INT, 0, comm);
    if (rank == 0) {
        free(display);
        free(scounts);
    }
}


/**
 * Execute la circulation sur les processeurs , transférant au processeur précédent
 * sa partie de la matrice et reçoit une partie de la matrice du processeur
 * suivant.
 * @param A bout de matrice stagnante
 * @param B bout de matrice tournante, elle sera réalloué si necessaire 
 * @param C matrice ou les résultats sont enregistrés
 * @param N nombre de ligne de la matrice B (ou colonne de A)
 * @param P nombre de colonne de B
 * @param numprocs nombre de processeurs
 * @param rank rank du processeur
 * @param nbLineAPerProc tableau du nombre de ligne de A pour chaque processeur la case 0 correspondant au rank 0
 * @param nbLineBPerProc tableau du nombre de ligne de B pour chaque processeur la case 0 correspondant au rank 0
 * @return 
 */
void doCirculation(int * A, int * B, int * C, int N, int P, int numprocs, int rank, int * nbLineAPerProc, int * nbLineBPerProc) {
    int i;
    int done = 0;
    /*
     * Calcul du decalage propre à chacun
     */
    for (i = 0; i < rank; i++) {
        done += nbLineBPerProc[i] % N;
    }
    int tour;
    for (tour = 0; tour < numprocs; tour++) {
        MPI_Status status;
        /*
         * Execution du calcul par tous les processeurs
         */
        calcul(A, B, C, nbLineAPerProc[rank], N, nbLineBPerProc[(tour + rank) % numprocs], P, done, shortestPath, rank);
        if (rank == 0) {
            MPI_Send(B, nbLineBPerProc[tour] * P, MPI_INT, numprocs - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(B, nbLineBPerProc[(tour + 1) % numprocs] * P, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);

        } else {
            int * temp = createMatriceLinearise(nbLineBPerProc[(tour + rank + 1) % numprocs], P);
            MPI_Recv(temp, nbLineBPerProc[(tour + rank + 1) % numprocs] * P, MPI_INT, (rank + 1) % numprocs, 0, MPI_COMM_WORLD, &status);

            MPI_Send(B, nbLineBPerProc[(tour + rank) % numprocs] * P, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
            /*
             * on passe du temporaire au réél.
             */
            int i;
            for (i = 0; i < nbLineBPerProc[(tour + rank + 1) % numprocs] * P; i++) {
                B[i] = temp[i];
            }
            free(temp);
        }
        /*
         * Incrémentation de la valeur de décalage
         */
        done += nbLineBPerProc[(tour + rank) % numprocs] % N;
    }
}