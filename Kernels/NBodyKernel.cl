double4 compute_acceleration(double4 position1, double4 position2, double mass2)
{
    double4 difference = position2 - position1;
    double distance_squared = difference.x * difference.x + difference.y * difference.y + difference.z * difference.z;
    
    // Stop OpenCL from crashing my fucking computer for the 4th time in a row.
    const double epsilon = 1e-10;
    if (distance_squared < epsilon) distance_squared = epsilon;

    // Compute acceleration, mass precomputed with G
    return mass2 / (distance_squared * sqrt(distance_squared)) * difference;
}

double4 calculate_total_acceleration(
    __global double4 *positions, __global double4 *velocities, __global double *masses,
    int numberOfBodies, int id
)
{
    double4 acceleration = (double4)(0.0, 0.0, 0.0, 0.0);
    double4 position = positions[id];

    for (int i = 0; i < numberOfBodies; i++)
    {
        if (i != id)
        {
            acceleration += compute_acceleration(position, positions[i], masses[i]);
        }
    }

    return acceleration;
}

__kernel void integrate_euler(__global double4 *positions, __global double4 *velocities, __global double *masses,
    double dt, int numberOfBodies)
{
    int globalId = get_global_id(0);

    velocities[globalId] += calculate_total_acceleration(positions, velocities, masses, numberOfBodies, globalId) * dt;
    positions[globalId] += velocities[globalId] * dt;
}

__kernel void integrate_verlet(__global double4 *positions, __global double4 *velocities, __global double *masses,
    double dt, int numberOfBodies)
{
    int globalId = get_global_id(0);
    
    double4 acceleration = calculate_total_acceleration(positions, velocities, masses, numberOfBodies, globalId);
    positions[globalId] += dt * (velocities[globalId] + acceleration * dt / 2);
    double4 new_acceleration = calculate_total_acceleration(positions, velocities, masses, numberOfBodies, globalId);
    velocities[globalId] += dt * (acceleration + new_acceleration) / 2;
}