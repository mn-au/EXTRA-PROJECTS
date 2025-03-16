#include <iostream>
#include <string>
#include <random>
#include <Eigen/Dense>

#define MAX_NUM 2014
#define MIN .12

double x_val[MAX_NUM];
double y_val[MAX_NUM];

// Declare the random number generator globally to avoid passing it every time
std::mt19937 generator;  // Global random number generator

// Function to initialize the random number generator once
void initialize_rng() {
    std::random_device rd;
    generator.seed(rd());
}

// Function to generate noisy data based on a quadratic curve
void make_data(int num) {
    if (num > MAX_NUM) return;

    std::uniform_real_distribution<double> distribution_x(-5, 5);  // Range for x values
    std::normal_distribution<double> distribution_y(0, 1);         // Noise for y values

    for (int i = 0; i < num; ++i) {
        x_val[i] = distribution_x(generator);
        y_val[i] = -(x_val[i] - MIN) * (x_val[i] - MIN) + distribution_y(generator);
    }
}

// Function to print data (optional)
void print_data(const char* f_name, int num) {
    FILE* out_file = fopen(f_name, "w");
    for (int i = 0; i < num; i++) {
        fprintf(out_file, "%f\t%f\n", x_val[i], y_val[i]);
    }
    fclose(out_file);  // Close the file
}

// Naive method: find the largest y value and return the corresponding x value
double find_max_naive(double* x, double* y, int num) {
    double max_x = x[0], max_y = y[0];
    for (int i = 1; i < num; i++) {
        if (y[i] > max_y) {
            max_y = y[i];
            max_x = x[i];
        }
    }
    return max_x;
}

// LLS method: fit y = a*x^2 + b*x + c using quadratic least squares and return the x value at the maximum
double find_max(double* x, double* y, int num) {
    Eigen::VectorXd y_vec(num);   // Vector for y values
    Eigen::MatrixXd X(num, 3);    // Design matrix for [x^2, x, 1]

    // Fill the design matrix and y vector
    for (int i = 0; i < num; i++) {
        y_vec[i] = y[i];
        X(i, 0) = x[i] * x[i];    // x^2 term
        X(i, 1) = x[i];           // x term
        X(i, 2) = 1;              // Constant term
    }

    // Solve for beta (the coefficients [a, b, c]) using OLS
    Eigen::Vector3d beta = (X.transpose() * X).inverse() * X.transpose() * y_vec;

    double a = beta(0);  // Coefficient for x^2
    double b = beta(1);  // Coefficient for x

    // Return the x value corresponding to the maximum of the quadratic fit (-b / 2a)
    return -b / (2 * a);
}

int main(void) {
    initialize_rng();  // Initialize the random number generator once

    int num_runs = 10;
    for (int i = 1; i <= num_runs; i++) {
        printf("Run number %d\n", i);

        // Generate new random data
        make_data(256);
        // Optionally, print data only once if needed
        if (i == 1) {
            print_data("DATA", 256);  // You can comment this out if file writing is not needed for every run
        }

        // Find the max values using both methods
        const double n_max = find_max_naive(x_val, y_val, 256);
        const double a_max = find_max(x_val, y_val, 256);

        // Output the results
        printf("TRUE MAX %f  Naive: %f  New Max: %f\n", MIN, n_max, a_max);
    }

    return 0;
}
