#include <iostream>
#include <time.h>
#include "omp.h"
#include <ctime>

// Как у баркалова
/*
void LU_Decomposition(double* A, double* L, double* U, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
        {
            U[n * i + j] = A[n * i + j];
        }
    }
    for (int i = 0; i < n; i++) {
        L[i * n + i] = 1;
        for (int k = i + 1; k < n; k++) {
            double mu = U[k * n + i] / U[n * i + i];
            for (int j = i; j < n; j++) {
                U[k * n + j] -= mu * U[i * n + j];
            }
            L[k * n + i] = mu;
            L[i * n + k] = 0;
        }
    }
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            U[i * n + j] = 0;
        }
    }
}
*/


// Блочное разложение
/*
void LU_Decomposition(double* A, double* L, double* U, int n, int r) {
    if (n <= r) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                U[n * i + j] = A[n * i + j];
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i * n + i] = 1;

            for (int k = i + 1; k < n; ++k) {
                double mu = U[k * n + i] / U[n * i + i];
                for (int j = i; j < n; ++j) {
                    U[k * n + j] -= mu * U[i * n + j];
                }
                L[k * n + i] = mu;
                L[i * n + k] = 0;
            }
        }
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                U[i * n + j] = 0;
            }
        }
    } else {

        int l = n - r;
        double* A11 = new double[r * r];
        double* A12 = new double[r * l];
        double* A21 = new double[r * l];


        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < r; ++j) {
                A11[i * r + j] = A[i * n + j];
            }
        }

        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < l; ++j) {
                A12[i * l + j] = A[i * n + j + r];
            }
        }

        for (int i = 0; i < l; ++i) {
            for (int j = 0; j < r; ++j) {
                A21[i * r + j] = A[(i + r) * n + j];
            }
        }

        double* L11 = new double[r * r];
        double* U11 = new double[r * r];

        LU_Decomposition(A11, L11, U11, r, r);

        delete[] A11;

        double* L21 = new double[r * l];
        double* U12 = new double[r * l];


        for (int iter = 0; iter < l; ++iter) {
            for (int i = 0; i < r; ++i) {
                L21[r * iter + i] = A21[r * iter + i];
                for (int j = 0; j < i; ++j) {
                    L21[r * iter + i] -= U11[r * j + i] * L21[r * iter + j];
                }
                L21[r * iter + i] /= U11[r * i + i];
            }
        }

        for (int iter = 0; iter < l; ++iter) {
            for (int i = 0; i < r; ++i) {
                U12[l * i + iter] = A12[l * i + iter];
                for (int j = 0; j < i; ++j) {
                    U12[l * i + iter] -= L11[r * i + j] * U12[l * j + iter];
                }
                U12[l * i + iter] /= L11[r * i + i];
            }
        }

        delete[] A12;
        delete[] A21;


        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < r; ++j) {
                L[i * n + j] = L11[i * r + j];
                U[i * n + j] = U11[i * r + j];
            }
        }
        delete[] L11;
        delete[] U11;

        double* A22 = new double[l * l];
        double* L22 = new double[l * l];
        double* U22 = new double[l * l];


        for (int i = 0; i < l; ++i) {
            for (int j = 0; j < l; ++j) {
                A22[i * l + j] = A[(i + r) * n + j + r];

                for (int k = 0; k < r; ++k) {
                    A22[i * l + j] -= L21[i * r + k] * U12[k * l + j];
                }
            }
        }
        LU_Decomposition(A22, L22, U22, l, r);

        delete[] A22;


        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < l; ++j) {
                L[i * n + j + r] = 0;
                U[i * n + j + r] = U12[i * l + j];
            }
        }

        for (int i = 0; i < l; ++i) {
            for (int j = 0; j < r; ++j) {
                L[(i + r) * n + j] = L21[i * r + j];
                U[(i + r) * n + j] = 0;
            }
        }

        for (int i = 0; i < l; ++i) {
            for (int j = 0; j < l; ++j) {
                L[(i + r) * n + j + r] = L22[i * l + j];
                U[(i + r) * n + j + r] = U22[i * l + j];
            }
        }

        delete[] L21;
        delete[] L22;

        delete[] U12;
        delete[] U22;
    }
}
*/



// Блочное разложение на потоки
void LU_Decomposition(double* A, double* L, double* U, int n, int r, int num_thread) {
    if (n <= r) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                U[n * i + j] = A[n * i + j];
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i * n + i] = 1;
#pragma omp parallel for num_threads(num_thread)
            for (int k = i + 1; k < n; ++k) {
                double mu = U[k * n + i] / U[n * i + i];
                for (int j = i; j < n; ++j) {
                    U[k * n + j] -= mu * U[i * n + j];
                }
                L[k * n + i] = mu;
                L[i * n + k] = 0;
            }
        }
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                U[i * n + j] = 0;
            }
        }
    } else {

        int l = n - r;
        double* A11 = new double[r * r];
        double* A12 = new double[r * l];
        double* A21 = new double[r * l];
        
#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < r; ++j) {
                A11[i * r + j] = A[i * n + j];
            }
        }
#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < l; ++j) {
                A12[i * l + j] = A[i * n + j + r];
            }
        }
#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < l; ++i) {
            for (int j = 0; j < r; ++j) {
                A21[i * r + j] = A[(i + r) * n + j];
            }
        }

        double* L11 = new double[r * r];
        double* U11 = new double[r * r];

        LU_Decomposition(A11, L11, U11, r, r, num_thread);

        delete[] A11;

        double* L21 = new double[r * l];
        double* U12 = new double[r * l];

#pragma omp parallel for num_threads(num_thread)
        for (int iter = 0; iter < l; ++iter) {
            for (int i = 0; i < r; ++i) {
                L21[r * iter + i] = A21[r * iter + i];
                for (int j = 0; j < i; ++j) {
                    L21[r * iter + i] -= U11[r * j + i] * L21[r * iter + j];
                }
                L21[r * iter + i] /= U11[r * i + i];
            }
        }
#pragma omp parallel for num_threads(num_thread)
        for (int iter = 0; iter < l; ++iter) {
            for (int i = 0; i < r; ++i) {
                U12[l * i + iter] = A12[l * i + iter];
                for (int j = 0; j < i; ++j) {
                    U12[l * i + iter] -= L11[r * i + j] * U12[l * j + iter];
                }
                U12[l * i + iter] /= L11[r * i + i];
            }
        }

        delete[] A12;
        delete[] A21;

#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < r; ++j) {
                L[i * n + j] = L11[i * r + j];
                U[i * n + j] = U11[i * r + j];
            }
        }
        delete[] L11;
        delete[] U11;

        double* A22 = new double[l * l];
        double* L22 = new double[l * l];
        double* U22 = new double[l * l];

#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < l; ++i) {
            for (int j = 0; j < l; ++j) {
                A22[i * l + j] = A[(i + r) * n + j + r];

                for (int k = 0; k < r; ++k) {
                    A22[i * l + j] -= L21[i * r + k] * U12[k * l + j];
                }
            }
        }
        LU_Decomposition(A22, L22, U22, l, r, num_thread);

        delete[] A22;

#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < l; ++j) {
                L[i * n + j + r] = 0;
                U[i * n + j + r] = U12[i * l + j];
            }
        }
#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < l; ++i) {
            for (int j = 0; j < r; ++j) {
                L[(i + r) * n + j] = L21[i * r + j];
                U[(i + r) * n + j] = 0;
            }
        }
#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < l; ++i) {
            for (int j = 0; j < l; ++j) {
                L[(i + r) * n + j + r] = L22[i * l + j];
                U[(i + r) * n + j + r] = U22[i * l + j];
            }
        }

        delete[] L21;
        delete[] L22;

        delete[] U12;
        delete[] U22;
    }
}

#define scalar(row, col) ((col) + (row) * size)
void Check_correct(double* A, double* L, double* U, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j)
        {
            double sum = 0;
            for (int k = 0; k < size; ++k)
                sum += L[scalar(i, k)] * U[scalar(k, j)];
            if (abs(A[scalar(i, j)] - sum) <= 0.01) {
                continue;
            }
            else {
                std::cout << "no correct" << "\n";
            }
        }
    }
    std::cout << "correct!" << "\n";
}


int main(int argc, char** argv) {
    const int n = 1000;
    const int r = 100;
    const int num_thread = 4;
    double* a = new double[n*n];
    double* l = new double[n*n];
    double* u = new double[n*n];
    double* As = new double[n*n];

    time_t begin, end;

    for (int i = 0; i < n*n; i++) {
        a[i] = rand() % 10 + 1;
    }
  

    begin = clock();
    LU_Decomposition(a, l, u, n, r, num_thread);
    end = clock();

    std::cout << "Time " << (end - begin) / 1000.0 << " sec." << "\n";

    //for (int i = 0; i < n; i++) {
    //    for (int j = 0; j < n; j++) {
    //        std::cout << a[i*j] << " ";
    //    }
    //    std::cout << "\n";
    //}
    //for (int i = 0; i < n; i++) {
    //    for (int j = 0; j < n; j++) {
    //        As[i*j] = 0;
    //        for (int k = 0; k < n; k++) {
    //            As[i*j] += l[i*k] * u[k*j];
    //        }
    //    }
    //}
    //std::cout << "------------------------- \n";
    //for (int i = 0; i < n; i++) {
    //    for (int j = 0; j < n; j++) {
    //        std::cout << As[i * j] << " ";
    //    }
    //    std::cout << "\n";
    //}
    Check_correct(a, l, u, n);

    delete[] a;
    delete[] l;
    delete[] u;
    delete[] As;
    return 0;
}