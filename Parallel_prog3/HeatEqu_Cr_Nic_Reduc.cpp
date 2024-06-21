#include <iostream>
#include <cmath>
#include <vector>
#include <time.h>
#include "omp.h"
#include <ctime>


class heat_task {
public:
    double T = 100;
    double L = 1;
    int n = 32;
    int m = 32;
    double initial_condition(double x) {
        return sin(3.14 * x);
    }
    double left_condition(double t) {
        return 1;
    }
    double right_condition(double t) {
        return -1;
    }
    double f(double x, double t) {
        return cos(3.14 + x);
    }
};

void heat_equation_crank_nicolson(heat_task task, double* v, int num_threads) {
    std::vector<double> u(task.n + 1);
    double dx = task.L / task.n;
    double dt = task.T / task.m;
    int log_n = std::log2(task.n);

//#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i <= task.n; ++i) {
        u[i] = task.initial_condition(i * dx);
    }

    double alpha = dt / (2 * dx * dx);
    double beta = -1 - dt / (dx * dx);
    double gamma1 = dt / (dx * dx) - 1;
    double gamma2 = dt / (2 * dx * dx);

    std::vector<double> a(log_n + 1);
    std::vector<double> b(log_n + 1);
    std::vector<double> c(log_n + 1);
    std::vector<double> rhs(task.n + 1);

    for (int j = 1; j <= task.m; ++j) {
//#pragma omp parallel for num_threads(num_threads)
        for (int i = 1; i < task.n; ++i) {
            rhs[i] = gamma1 * u[i] - gamma2 * (u[i - 1] + u[i + 1]) -
            dt * task.f(i * dx, (j - 0.5) * dt);
        }
        rhs[1] -= task.left_condition(j * dt) * (dt / (dx * dx * 2));
        rhs[task.n - 1] -= task.right_condition(j * dt) * (dt / (dx * dx * 2));

        a[0] = alpha;
        b[0] = beta;
        c[0] = alpha;

        rhs[0] = 0;
        rhs[task.n] = 0;
        u[0] = 0;
        u[task.n] = 0;

        int start = 2;
        int size_n = task.n;
        int step = 1;
        for (int j = 0; j < log_n; ++j) {
            double a_k = -a[j] / b[j];
            double c_k = -c[j] / b[j];
            a[j + 1] = a_k * a[j];
            b[j + 1] = b[j] + a_k * a[j] + c_k * c[j];
            c[j + 1] = c_k * c[j];

            size_n = (size_n - 1) / 2;
//#pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < size_n; ++i) {
                int idx = start * (i + 1);
                rhs[idx] = a_k * rhs[idx - step] + rhs[idx] + c_k * rhs[idx + step];
            }
            start = 2 * start;
            step = 2 * step;
        }

        start = task.n / 2;
        size_n = 1;
        for (int k = log_n - 1; k >= 0; --k) {
            double a_k = -a[k] / b[k];
            double c_k = -c[k] / b[k];
//#pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < size_n; ++i) {
                int idx = start * (2 * i + 1);
                u[idx] = rhs[idx] / b[k] + a_k * u[idx - start] + c_k * u[idx + start];
            }
            start = start / 2;
            size_n = size_n * 2;
        }

        u[0] = task.left_condition(j * dt);
        u[task.n] = task.right_condition(j * dt);
    }

//#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i <= task.n; ++i) {
        v[i] = u[i];
    }
}


int main(int argc, char* argv[]) {
    heat_task t;
    const int z = 8192;
    //const int z = 16384;
    //const int z = 32768;
    //const int z = 65536;
    const int num_thread = 1;
    t.n = z;
    t.m = z;
    double* v = new double[(t.n + 1) * (t.m + 1)];
    time_t begin, end;

    begin = clock();
    heat_equation_crank_nicolson(t, v, num_thread);
    end = clock();

    std::cout << "Time " << (end - begin) / 1000.0 << " sec." << "\n";

    delete[] v;
    return 0;
}