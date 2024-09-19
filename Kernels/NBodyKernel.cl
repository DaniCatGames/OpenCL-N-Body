double4 compute_acceleration(double4 position1, double4 position2, double mass2) {
    double4 difference = position2 - position1;
    double distance_squared = difference.x * difference.x + difference.y * difference.y + difference.z * difference.z + 1e-10;

    // Compute acceleration, mass precomputed with G
    return mass2 / (distance_squared * sqrt(distance_squared)) * difference;
}

double4 calculate_total_acceleration(
    __global double4 *positions, __global double *masses,
    int numberOfBodies, int id) {

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

double4 calculate_total_acceleration_with_position(
    __global double4 *positions, __global double *masses,
    int numberOfBodies, int id, double4 position) {

    double4 acceleration = (double4)(0.0, 0.0, 0.0, 0.0);

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
    double dt, int numberOfBodies) {

    int globalId = get_global_id(0);

    velocities[globalId] += calculate_total_acceleration(positions, masses, numberOfBodies, globalId) * dt;
    positions[globalId] += velocities[globalId] * dt;
}

__kernel void integrate_verlet(__global double4 *positions, __global double4 *velocities, __global double *masses,
    double dt, int numberOfBodies) {

    int globalId = get_global_id(0);
    
    double4 acceleration = calculate_total_acceleration(positions, masses, numberOfBodies, globalId);
    positions[globalId] += dt * (velocities[globalId] + acceleration * dt / 2);
    double4 new_acceleration = calculate_total_acceleration(positions, masses, numberOfBodies, globalId);
    velocities[globalId] += dt * (acceleration + new_acceleration) / 2;
}

__kernel void integrate_rk4(__global double4 *positions, __global double4 *velocities, __global double *masses,
    double dt, int numberOfBodies) {
    
    int globalId = get_global_id(0);
    
    double4 position = positions[globalId];
    double4 velocity = velocities[globalId];
    double half_dt = dt / 2;
    
    //K1 stage
    double4 k1_position = velocity;
    double4 k1_velocity = calculate_total_acceleration(positions, masses, numberOfBodies, globalId);
    
    //K2 stage
    double4 temporary_position = position + half_dt * k1_position;
    double4 temporary_velocity = velocity + half_dt * k1_velocity;
    double4 k2_position = temporary_velocity;
    double4 k2_velocity = calculate_total_acceleration_with_position(positions, masses, numberOfBodies, globalId, temporary_position);
    
    //K3 stage
    temporary_position = position + half_dt * k2_position;
    temporary_velocity = velocity + half_dt * k2_velocity;
    double4 k3_position = temporary_velocity;
    double4 k3_velocity = calculate_total_acceleration_with_position(positions, masses, numberOfBodies, globalId, temporary_position);
        
    //K4 stage
    temporary_position = position + half_dt * k3_position;
    temporary_velocity = velocity + half_dt * k3_velocity;
    double4 k4_position = temporary_velocity;
    double4 k4_velocity = calculate_total_acceleration_with_position(positions, masses, numberOfBodies, globalId, temporary_position);
    
    //Update
    positions[globalId] += (k1_position + 2 * k2_position + 2 * k3_position + k4_position) * (half_dt / 3);
    velocities[globalId] += (k1_velocity + 2 * k2_velocity + 2 * k3_velocity + k4_velocity) * (half_dt / 3);
}