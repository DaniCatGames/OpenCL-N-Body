kernel void nbody_step(
    global double *positions,
    global double *velocities,
    global double *masses
    ) {
    
    const int numberOfBodies = 2;
    const int timestep = 1;
    
    const int index = get_global_id(0);
    const int xIndex = index * 3;
    const int yIndex = xIndex + 1;
    const int zIndex = xIndex + 2;
    
    double xAcc = 0;
    double yAcc = 0;
    double zAcc = 0;

    if (index < numberOfBodies) {
        double xPos = positions[xIndex];
        double yPos = positions[yIndex];
        double zPos = positions[zIndex];
        
        for (int i = 0; i < numberOfBodies; i++) {
            if (i == index) { // skip self
                continue;
            }
            
            // distance
            double xDis = xPos - positions[i * 3];
            double yDis = yPos - positions[i * 3 + 1];
            double zDis = zPos - positions[i * 3 + 2];
            
            // squared distance is needed twice
            double squareDistance = xDis * xDis + yDis * yDis + zDis * zDis;
            
            // masses precalculated with G=6.67408*10^(-11)
            double gravForce = -native_divide(masses[i], squareDistance) * native_rsqrt(squareDistance);
            // (G*Mi)/(r^3)
            
            //calculate the acceleration
            xAcc += xDis * gravForce; // a = (G*Mi*xR)/(r^3)
            yAcc += yDis * gravForce;
            zAcc += zDis * gravForce;
        }

        velocities[xIndex] += xAcc * timestep;
        velocities[yIndex] += yAcc * timestep;
        velocities[zIndex] += zAcc * timestep;

        positions[xIndex] += velocities[xIndex] * timestep;
        positions[yIndex] += velocities[yIndex] * timestep;
        positions[zIndex] += velocities[zIndex] * timestep;
    }
}