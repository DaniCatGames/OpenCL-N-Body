double squareDistance(double3 position1, double3 position2) {
    double3 difference = position2 - position1;
    double distance = difference.x * difference.x + difference.y * difference.y + difference.z * difference.z;
    return distance;
}

double3 compute_acceleration(double3 position1, double3 position2, double mass2) {
    double3 difference = position2 - position1;
    double distanceSquared = squareDistance(position1, position2);
    const double epsilon = 1e-10;
    
    // Avoid division by zero
    if (distanceSquared < epsilon) {
        distanceSquared = epsilon;
    }

    // Compute acceleration
    return mass2 / (distanceSquared * sqrt(distanceSquared)) * difference;
}

__kernel void nbody_step(
    __global double *positions,
    __global double *velocities,
    __global double *masses    
) {
    const double dt = 0.001; // Adjusted time step for better accuracy
    const int numberOfBodies = 2;

    int globalId = get_global_id(0);
    double3 acceleration = (double3)(0.0, 0.0, 0.0);
    double3 position = vload3(globalId, *positions);

    for (int i = 0; i < numberOfBodies; i++) {
        if (i != globalId) {
            acceleration += compute_acceleration(position, vload3(i, positions), masses[i]);
        }
    }

    // Update velocity and position
    double3 velocity = vload3(globalId, velocity) + acceleration * dt;
    position += velocity * dt;
    
    vstore3(velocity, globalId, *velocities);
    vstore3(position, globalId, *positions);
}