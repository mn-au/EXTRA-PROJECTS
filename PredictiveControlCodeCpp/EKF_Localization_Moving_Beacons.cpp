/*
  * Extended Kalman Filter (EKF) Localization with Moving Beacons
  * Project Name: EKF_Localization_Moving_Beacons
  * 
  * Description: 
  * - Implements Extended Kalman Filter (EKF) for robot localization.
  * - Uses moving satellites (beacons) for distance-only updates.
  * - Includes motion model, noise simulation, and localization updates.
  */
 

 #include <iostream>
 #include <Eigen/Dense>
 #include <math.h>
 #include <random>
 
 // Time step for simulation
 #define DT 0.01
 
 // Toggle noise for realistic simulation
 #define NOISE
 
 #define M_PI 3.14159265358979323846
 
 // Formatting for Eigen output
 Eigen::IOFormat fmt(5, 1, ", ", " ", "", "", "", "");
 
 //-------------------------------------------------------
 // Helper Functions
 
 // Normalize an angle to the range [-π, π]
 double normal_angle(double x)
 {
     if (x > M_PI) return (x - 2. * M_PI);
     if (x < -M_PI) return (x + 2. * M_PI);
     return x;
 }
 
 //-------------------------------------------------------
 // Beacons (Markers) Implementation
 #define MAX_NUM_B 32
 
 class Marker {
 public:
     double px, py; // Position of the marker
 };
 
 Marker markers[MAX_NUM_B];
 int num_markers = 0;
 
 void add_marker(double x, double y) {
     if (num_markers >= MAX_NUM_B) return;
     markers[num_markers].px = x;
     markers[num_markers].py = y;
     num_markers++;
 }
 
 void print_markers(void) {
     FILE *f_out = fopen("MARK", "w");
     for (int i = 0; i < num_markers; i++) {
         printf("Marker %d: x=%f, y=%f\n", i, markers[i].px, markers[i].py);
         fprintf(f_out, "%f\t%f\n", markers[i].px, markers[i].py);
     }
     fclose(f_out);
 }
 
 //-------------------------------------------------------
 // Satellite Class (Moving Beacons)
 #define SAT_SPEED 3  // Satellite movement speed
 
 class Satellite {
 public:
     Satellite(double px, double py, double bearing);
     void Move(double dt);
     double getDistance(double x, double y);
     void Print(FILE *file);
     double getX(void) { return px; }
     double getY(void) { return py; }
 
 private:
     double px, py, theta; // Position and direction
 };
 
 Satellite::Satellite(double _px, double _py, double bearing) {
     px = _px;
     py = _py;
     theta = bearing;
 }
 
 void Satellite::Move(double dt) {
     px += dt * SAT_SPEED * cos(theta);
     py += dt * SAT_SPEED * sin(theta);
 }
 
 double Satellite::getDistance(double x, double y) {
     return sqrt((px - x) * (px - x) + (py - y) * (py - y));
 }
 
 void Satellite::Print(FILE *file) {
     fprintf(file, "%f\t%f\t", px, py);
 }
 
 //-------------------------------------------------------
 // Satellite Manager
 Satellite *satellites[MAX_NUM_B];
 int num_satellites = 0;
 
 void add_satellite(double x, double y, double bearing) {
     if (num_satellites >= MAX_NUM_B) return;
     satellites[num_satellites] = new Satellite(x, y, bearing);
     num_satellites++;
 }
 
 void move_satellites(void) {
     for (int i = 0; i < num_satellites; i++) {
         satellites[i]->Move(DT);
     }
 }
 
 void print_satellites(FILE *file) {
     for (int i = 0; i < num_satellites; i++) {
         satellites[i]->Print(file);
     }
     fprintf(file, "\n");
 }
 
 //-------------------------------------------------------
 // Robot Class (Extended Kalman Filter Implementation)
 class Robot {
 public:
     Robot(double _dt);
     ~Robot() {}
 
     void Move();
     Eigen::VectorXd GetState() { return X; }
     void LocalizeM(); // Marker localization (angle + distance)
     void LocalizeS(); // Satellite localization (distance-only)
     void LocalizeC(); // Compass localization (direction-only)
 
     void Print();
     void PrintVariance();
 
     void JacobiF(Eigen::MatrixXd &F, const Eigen::VectorXd &X);
     void JacobiV(Eigen::MatrixXd &V, const Eigen::VectorXd &X);
     void JacobiH(Eigen::MatrixXd &H, const Eigen::VectorXd &X, double px, double py);
     void JacobiHD(Eigen::MatrixXd &H, const Eigen::VectorXd &X, double px, double py); // Distance-only Jacobian
 
 private:
     Eigen::VectorXd X; // Estimated state (x, y, theta)
     Eigen::MatrixXd P; // Covariance matrix
 
     double px, py, theta; // Actual position and orientation
     double steer, speed, dt; // Control inputs
 
     Eigen::MatrixXd R, Rc, Rs, M; // Measurement & control noise matrices
     Eigen::MatrixXd JF, JV, JH, JHD; // Jacobian matrices
 
     FILE *s_file; // File for logging robot state
     FILE *l_file; // File for localization logging
 };
 
 //-------------------------------------------------------
 // Initialization Function
 void init_setup(void) {
     num_markers = 0;
     num_satellites = 0;
     
     // Moving beacon example
     add_satellite(5, -10, M_PI * 0.5);
     print_markers();
 }
 
 //-------------------------------------------------------
 // Main Function
 #define NUM_ITER 200
 
 int main(void) {
     init_setup();
     FILE *sat_file = fopen("SAT", "w"); // Satellite logging file
     
     double time = 0;
     Robot rob(DT);
 
     while (time < 10.) {
         move_satellites();
         rob.Move();
 
         if ((int)(time / DT) % 10 == 0) { // Localize every 10 iterations
             print_satellites(sat_file);
             rob.LocalizeS(); // Distance-only localization
             rob.PrintVariance();
         }
         rob.Print();
         time += DT;
     }
     fclose(sat_file);
 }
 