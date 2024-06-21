#include <iostream>
#include <vector>
#include <time.h>
#include "omp.h"

struct CRSMatrix {
    int n = 0;
    int m = 0;
    int nz = 0;
    std::vector<double> val;
    std::vector<int> colIndex;
    std::vector<int> rowPtr;
};


double scalar_product(double* a, double* b, int n, int num_thread) {
    double result = 0;
//#pragma omp parallel for num_threads(num_thread) reduction(+: result)
    for (int i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}


void matrix_multiplication(CRSMatrix& A, double* b, double* mul, int num_thread) {
//#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < A.n; ++i) {
        mul[i] = 0.0;
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            mul[i] += A.val[j] * b[A.colIndex[j]];
        }
    }
}


void SLE_Solver_CRS_BICG(CRSMatrix& A, double* b, double eps, int max_iter, double* x, int & count, int num_thread) {
    int n_size = A.n;
    CRSMatrix AT;

    AT.n = A.m;
    AT.m = A.n;

    std::vector<std::vector<int>> index(AT.n);
    std::vector<std::vector<double>> value(AT.n);
    std::vector<int> size(AT.n);

    for (int i = 0; i < A.n; ++i) {
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            index[A.colIndex[j]].push_back(i);
            value[A.colIndex[j]].push_back(A.val[j]);
            size[A.colIndex[j]] += 1;
        }
    }

    AT.rowPtr.push_back(0);
    for (int i = 0; i < AT.n; ++i) {
        AT.rowPtr.push_back(AT.rowPtr[i] + size[i]);
        for (int j = 0; j < size[i]; ++j) {
            AT.val.push_back(value[i][j]);
            AT.colIndex.push_back(index[i][j]);
        }
    }

//#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < n_size; ++i) {
        x[i] = 1;
    }

    double* Ap = new double[n_size];
    double* r = new double[n_size];
    double* r_ = new double[n_size];

    double* p = new double[n_size];
    double* p_ = new double[n_size];

    matrix_multiplication(A, x, Ap, num_thread);

//#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < n_size; ++i) {
        r[i] = b[i] - Ap[i];
        r_[i] = r[i];
        p[i] = r[i];
        p_[i] = r_[i];
    }

    double* Ap_ = new double[n_size];

    double* r_next = new double[n_size];
    double* r_next_ = new double[n_size];

    double alpha = 1;
    double betta = 1;

    double norm = sqrt(scalar_product(b, b, n_size, num_thread));

    while (true) {
        count++;
        if (sqrt(scalar_product(r, r, n_size, num_thread)) / norm < eps) {
            std::cout << "Scalar" << "\n";
            break;
        }
        if (count >= max_iter) {
            std::cout << "Iter" << "\n";
            break;
        }
        //if (abs(betta) < 1e-6) {
        //    std::cout << "Betta" << "\n";
        //    break;
        //}

        matrix_multiplication(A, p, Ap, num_thread);
        matrix_multiplication(AT, p_, Ap_, num_thread);
        double scalar_tmp = scalar_product(r, r_, n_size, num_thread);
        alpha = scalar_tmp / scalar_product(Ap, p_, n_size, num_thread);

//#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < n_size; ++i) {
            x[i] += alpha * p[i];
            r_next[i] = r[i] - alpha * Ap[i];
            r_next_[i] = r_[i] - alpha * Ap_[i];
        }

        betta = scalar_product(r_next, r_next_, n_size, num_thread) / scalar_tmp;

//#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < n_size; ++i) {
            p[i] = r_next[i] + betta * p[i];
            p_[i] = r_next_[i] + betta * p_[i];
            r[i] = r_next[i];
            r_[i] = r_next_[i];
        }
    }
    delete[] Ap;
    delete[] Ap_;

    delete[] r;
    delete[] r_;

    delete[] r_next;
    delete[] r_next_;

    delete[] p;
    delete[] p_;

}


void generateSparseMatrix(int rows, int cols, double sparsity, std::vector<std::vector<double>>& A) {
    for (int i = 0; i < rows; ++i) {
        std::vector<double> row;
        for (int j = 0; j < cols; ++j) {
            double elem = (double)rand() / RAND_MAX;
            if (elem > sparsity) {
                row.push_back(elem);
            }
            else {
                row.push_back(0);
            }
        }
        A.push_back(row);
    }
}


void convertToCRS(const std::vector<std::vector<double>>& A, CRSMatrix& crsA) {
    crsA.rowPtr.push_back(0);

    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[i].size(); ++j) {
            if (A[i][j] != 0.0) {
                crsA.val.push_back(A[i][j]);
                crsA.colIndex.push_back(j);
            }
        }
        crsA.rowPtr.push_back(crsA.val.size());
    }
}


int main(int argc, char* argv[]) 
{   
    srand(33);
    const int num_thread = 1;
    const int n = 5000;
    const int m = 5000;
    double eps = 0.00001;
    int N_iter = 10000;
    int count = 0;

    time_t begin, end;
    std::vector<std::vector<double>> A_old;
    CRSMatrix A;

    double* b = new double[n * sizeof(double)];
    double* x = new double[n * sizeof(double)];


    for (int i = 0; i < n; ++i) {
        double elem = rand() % 1000 + 10;
        x[i] = elem;
    }

    generateSparseMatrix(n, m, 0.95, A_old);

    for (int i = 0; i < n; i++) {
        double tmp = 0;
        for (int j = 0; j < n; j++) {
            tmp += A_old[i][j] * x[j];
        }
        b[i] = tmp;
    }

    A.n = n;
    A.m = m;
    convertToCRS(A_old, A);

    A_old = { {1}, {1} };
    begin = clock();
    SLE_Solver_CRS_BICG(A, b, eps, N_iter, &(*x), count, num_thread);
    end = clock();
    std::cout << "Time " << (end - begin) / 1000.0 << " sec." << "\n";

    return 0;
}