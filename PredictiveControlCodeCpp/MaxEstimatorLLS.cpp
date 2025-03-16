#include <iostream>
#include <Eigen/Dense>
#include <fstream>

using namespace Eigen;
using namespace std;

#define M_PI 3.14159265358979323846
 
// Continuous system dynamics: dx/dt = f(x, t)
void continuousSystem(double gravity, double velocity_x, double velocity_y, Vector4d& state, Vector4d& derivatives) {
    derivatives(0) = velocity_x;          // dx/dt = velocity_x
    derivatives(1) = 0;                   // dvx/dt = 0 (no force in x)
    derivatives(2) = velocity_y;          // dy/dt = velocity_y
    derivatives(3) = -gravity;            // dvy/dt = -g (gravity affects y velocity)
}

// Runge-Kutta 4th order method for continuous simulation
void simulateContinuousRK4(double timestep, int steps, double gravity, double initial_velocity, double angle, ofstream &outfile) {
    Vector4d state;
    state << 0, initial_velocity * cos(angle), 0, initial_velocity * sin(angle);  // Initial [pos_x, vel_x, pos_y, vel_y]

    // Runge-Kutta method (RK4) variables
    Vector4d k1, k2, k3, k4, temp_state;

    // Loop over the number of steps
    for (int step = 0; step < steps; ++step) {
        // Get velocities from current state
        double velocity_x = state(1);
        double velocity_y = state(3);

        // Calculate k1
        continuousSystem(gravity, velocity_x, velocity_y, state, k1);

        // Calculate k2
        temp_state = state + 0.5 * timestep * k1;
        continuousSystem(gravity, temp_state(1), temp_state(3), temp_state, k2);

        // Calculate k3
        temp_state = state + 0.5 * timestep * k2;
        continuousSystem(gravity, temp_state(1), temp_state(3), temp_state, k3);

        // Calculate k4
        temp_state = state + timestep * k3;
        continuousSystem(gravity, temp_state(1), temp_state(3), temp_state, k4);

        // Update the state using RK4
        state += (timestep / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);

        // Stop if the ball hits the ground
        if (state(2) <= 0) {
            outfile << state(0) << "," << state(2) << "\n";  // End of continuous data
            break;
        }

        // Log continuous x, y values to CSV
        outfile << state(0) << "," << state(2) << "\n";
    }
}

// Function to simulate the discrete system using matrix operations
void simulateDiscrete(double timestep, int steps, double gravity, double initial_velocity, double angle, ofstream &outfile) {
    // Initial state: [x_position, x_velocity, y_position, y_velocity]
    Vector4d state;
    state << 0, initial_velocity * cos(angle), 0, initial_velocity * sin(angle);  // Initial [pos_x, vel_x, pos_y, vel_y]

    // Discrete state matrix A_discrete
    Matrix4d A_discrete;
    A_discrete << 1, timestep, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, timestep,
                  0, 0, 0, 1;

    // Discrete input matrix B_discrete (gravity affects y-velocity)
    Vector4d B_discrete;
    B_discrete << 0, 0, 0, -timestep * gravity;

    // Simulate the system
    for (int step = 0; step < steps; ++step) {
        // Update the state vector using the discrete-time system
        state = A_discrete * state + B_discrete;

        // If the ball touches the ground (y <= 0), stop the simulation
        if (state(2) <= 0) {
            outfile << state(0) << "," << state(2) << "\n";  // Log discrete data
            break;
        }

        // Log discrete x, y values to CSV
        outfile << state(0) << "," << state(2) << "\n";
    }
}

int main() {
    double timestep = 0.01;  // Time step for both continuous and discrete simulations
    int steps = 1000;        // Number of simulation steps
    double gravity = 9.81;   // Gravitational constant
    double angle = M_PI / 4; // Launch angle (in radians)
    double initial_velocity = 20.0;  // Starting speed

    // Open files for writing trajectory data
    ofstream outfile_cont("continuous_trajectory.csv");
    ofstream outfile_disc("discrete_trajectory.csv");

    // Write headers for both files
    outfile_cont << "x_cont,y_cont\n";
    outfile_disc << "x_disc,y_disc\n";

    // Simulate the continuous system using Runge-Kutta
    simulateContinuousRK4(timestep, steps, gravity, initial_velocity, angle, outfile_cont);

    // Simulate the discrete system
    simulateDiscrete(timestep, steps, gravity, initial_velocity, angle, outfile_disc);

    // Close the files
    outfile_cont.close();
    outfile_disc.close();

    cout << "Simulation complete. Results written to continuous_trajectory.csv and discrete_trajectory.csv." << endl;

    return 0;
}
