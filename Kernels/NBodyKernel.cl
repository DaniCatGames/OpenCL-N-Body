double4 compute_acceleration(double4 position1, double4 position2, double mass2) {
    double4 difference = position2 - position1;
    double distanceSquared = difference.x * difference.x + difference.y * difference.y + difference.z * difference.z;
    const double epsilon = 1e-10;
    
    if (distanceSquared < epsilon) distanceSquared = epsilon;

    // Compute acceleration, mass precomputed with G
    return mass2 / (distanceSquared * sqrt(distanceSquared)) * difference;
}

__kernel void brute_force_calculations_haha_get_it(
    __global double4 *positions,
    __global double4 *velocities,
    __global double *masses,
    
    double dt,
    int numberOfBodies
) {
    int globalId = get_global_id(0);
    double4 acceleration = (double4)(0.0, 0.0, 0.0, 0.0);
    double4 position = positions[globalId];

    for (int i = 0; i < numberOfBodies; i++) {
        if (i != globalId) {
            acceleration += compute_acceleration(position, positions[i], masses[i]);
        }
    }

    // Update velocity and position
    velocities[globalId] += acceleration * dt;
    positions[globalId] += velocities[globalId] * dt;
}