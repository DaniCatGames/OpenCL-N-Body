namespace OpenCL_Barnes_Hut;

public enum UniverseSetups {
	EarthMoonSatellites,
	Infinity,
	LiLiaoUE2D,
	SunSystem,
}

public abstract class Universe {
	private const double G = 6.674315e-11;

	public static (double[] positions, double[] velocities, double[] masses) GetUniverse(int numberOfBodies,
		UniverseSetups universeSetup, IntegrationMethods integrationMethod, double timeStep = 0) {

		return universeSetup switch {
			UniverseSetups.EarthMoonSatellites => EarthMoonSatellites(numberOfBodies, timeStep, integrationMethod),
			UniverseSetups.SunSystem => SunSystem(timeStep, integrationMethod),
			_ => Periodic(timeStep, integrationMethod, universeSetup),
		};
	}

	// Create initial data
	private static (double[] positions, double[] accelerations, double[] masses) EarthMoonSatellites(int numberOfBodies,
		double timeStep, IntegrationMethods integrationMethod) {
		var positions = new Double4[numberOfBodies];
		var velocities = new Double4[numberOfBodies];
		var masses = new double[numberOfBodies];

		masses[0] = 5.97219e24 * G; // Mass Earth
		masses[1] = 7.348e22 * G; // Mass Moon 
		positions[0] = new Double4(0.0); //Position Earth
		positions[1] =
			new Double4(-3.679525311419403E+08, 1.665317952836706E+08, 2.517731761270612E+07); //Position Moon
		velocities[0] = new Double4(0.0);
		velocities[1] = new Double4(-4.097678496220849E+02, -8.756270441426266E+02, -5.926517055394465E+01);

		for (var i = 2; i < numberOfBodies; i++) {
			masses[i] = 1200;
			positions[i] = new Double4(6571000 + i * 1000, 0.0, 0.0);
			velocities[i] = new Double4(0.0, 7800 + 3386 * (i / (float)numberOfBodies), 0.0);
		}

		if (timeStep == 0) return (FlattenDoubleArray(positions), FlattenDoubleArray(velocities), masses);
		
		return integrationMethod switch {
			IntegrationMethods.M52 => M52Init(numberOfBodies, timeStep, positions, velocities, masses),
			IntegrationMethods.M157 => M157Init(numberOfBodies, timeStep, positions, velocities, masses),
			_ => M52Init(numberOfBodies, timeStep, positions, velocities, masses)
		};
	}
	
	private static (double[] positions, double[] accelerations, double[] masses) SunSystem(
		double timeStep, IntegrationMethods integrationMethod) {
		int numberOfBodies = 10;
		var positions = new Double4[numberOfBodies];
		var velocities = new Double4[numberOfBodies];
		var masses = new double[numberOfBodies];
		
		// Sol
		positions[0] = new Double4(0.0);
		velocities[0] = new Double4(0.0);
		masses[0] = 1.32712440041279419e20;
		
		// Mercury
		positions[1] = new Double4(-5.795101500995139e10, -2.433278906142044e10, 3.349520977838451e9);
		velocities[1] = new Double4(8.795954090184580e3, -4.277184806667985e4, -4.298000562200087e3);
		masses[1] = 2.2031868551e13;
		
		// Venus
		positions[2] = new Double4(1.046967023946389e11, -2.897419210863953e10, -6.439748486360615e9);
		velocities[2] = new Double4(9.157098341447236e3, 3.359680767896538e4, -7.949455890246071e1);
		masses[2] = 3.24858592e14;
		
		// Earth
		positions[3] = new Double4(-2.945365864135111e10, 1.441109954996951e11, 3.209088089852035e7);
		velocities[3] = new Double4(-2.967304298937015e4, -6.073752073217418e3, -1.989770692770065e0);
		masses[3] = 4.0351883398e14;
		
		// Mars
		positions[4] = new Double4(6.513001563190811e10, -2.023328602521687e11, -5.846243796763211e9);
		velocities[4] = new Double4(2.399445406036142e4, 9.493149473822923e3, -3.964138870980869e2);
		masses[4] = 4.2828375816e13;
		
		// Jupiter
		positions[5] = new Double4(-4.511930764410675e11, -6.672353530039495e11, 1.283616559709314e10);
		velocities[5] = new Double4(1.067887857676001e4, -6.708072196740888e3, -2.122805836201755e2);
		masses[5] = 1.267127641e17;
		
		// Saturn
		positions[6] = new Double4(-5.489745641419701e10, -1.504707985483393e12, 2.866054780789477e10);
		velocities[6] = new Double4(9.130281221794307e3, -3.944930245934130e2, -3.548036352631895e2);
		masses[6] = 3.79405848418e16;
		
		// Uranus
		positions[7] = new Double4(-9.692829961267297e11, -2.670837235275938e12, 2.657448000032902e9);
		velocities[7] = new Double4(6.361787628799833e3, -2.646449103674936e3, -9.272641933409409e1);
		masses[7] = 5.7945564e15;
		
		// Neptune
		positions[8] = new Double4(2.266223889656152e11, 4.461841849742382e12, -9.710614902790475e10);
		velocities[8] = new Double4(-5.447617894349518e3, 3.044797889906726e2, 1.187834180290304e2);
		masses[8] = 6.83652710058e15;
		
		// Pluto
		positions[9] = new Double4(1.540430119826912e12, 6.753703363101277e12, -1.168940890450712e12);
		velocities[9] = new Double4(-3.746243543279277e3, 3.710200115562642e2, 1.068327978773231e3);
		masses[9] = 9.755e11;

		if (timeStep == 0) return (FlattenDoubleArray(positions), FlattenDoubleArray(velocities), masses);
		
		return integrationMethod switch {
			IntegrationMethods.M52 => M52Init(numberOfBodies, timeStep, positions, velocities, masses),
			IntegrationMethods.M157 => M157Init(numberOfBodies, timeStep, positions, velocities, masses),
			_ => M52Init(numberOfBodies, timeStep, positions, velocities, masses)
		};
	}
	

	private static (double[] positions, double[] velocities, double[] masses) Periodic(double timeStep,
		IntegrationMethods integrationMethod, UniverseSetups universeSetup) {
		var numberOfBodies = 3;
		var positions = PeriodicOrbitData.PositionLookupTable()[universeSetup];
		var velocities = PeriodicOrbitData.VelocityLookupTable()[universeSetup];
		var masses = PeriodicOrbitData.MassLookupTable()[universeSetup];

		if (timeStep == 0) return (FlattenDoubleArray(positions), FlattenDoubleArray(velocities), masses);
		
		return integrationMethod switch {
			IntegrationMethods.M52 => M52Init(numberOfBodies, timeStep, positions, velocities, masses),
			IntegrationMethods.M157 => M157Init(numberOfBodies, timeStep, positions, velocities, masses),
			_ => M52Init(numberOfBodies, timeStep, positions, velocities, masses)
		};
	}
	
	
	// Multistep Initialization
	private static (double[] positions, double[] accelerations, double[] masses) M52Init(int numberOfBodies,
		double timeStep, Double4[] positions, Double4[] velocities, double[] masses) {
		//Calculate previous positions with Leapfrog using timesteps of dt/512
		var positions1 = CalculatePositionsAtTime(1, timeStep, positions, velocities, masses, numberOfBodies);
		var positions2 = CalculatePositionsAtTime(2, timeStep, positions, velocities, masses, numberOfBodies);
		var positions3 = CalculatePositionsAtTime(3, timeStep, positions, velocities, masses, numberOfBodies);

		var accelerations1 = new Double4[numberOfBodies];
		var accelerations2 = new Double4[numberOfBodies];

		for (var i = 0; i < numberOfBodies; i++)
			accelerations1[i] = CalculateTotalAcceleration(positions1, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations2[i] = CalculateTotalAcceleration(positions2, masses, numberOfBodies, i);

		var positionsArray = new Double4[numberOfBodies * 5];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i] = new Double4(0);
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 1] = positions[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 2] = positions1[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 3] = positions2[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 4] = positions3[i];

		var accelerationsArray = new Double4[numberOfBodies * 3];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i] = new Double4(0);
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 1] = accelerations1[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 2] = accelerations2[i];


		return (FlattenDoubleArray(positionsArray), FlattenDoubleArray(accelerationsArray), masses);
	}

	private static (double[] positions, double[] accelerations, double[] masses) M157Init(int numberOfBodies,
		double timeStep, Double4[] positions, Double4[] velocities, double[] masses) {
		//Calculate previous positions with Leapfrog using timesteps of dt/512
		var positions1 = CalculatePositionsAtTime(1, timeStep, positions, velocities, masses, numberOfBodies);
		var positions2 = CalculatePositionsAtTime(2, timeStep, positions, velocities, masses, numberOfBodies);
		var positions3 = CalculatePositionsAtTime(3, timeStep, positions, velocities, masses, numberOfBodies);
		var positions4 = CalculatePositionsAtTime(4, timeStep, positions, velocities, masses, numberOfBodies);
		var positions5 = CalculatePositionsAtTime(5, timeStep, positions, velocities, masses, numberOfBodies);
		var positions6 = CalculatePositionsAtTime(6, timeStep, positions, velocities, masses, numberOfBodies);
		var positions7 = CalculatePositionsAtTime(7, timeStep, positions, velocities, masses, numberOfBodies);
		var positions8 = CalculatePositionsAtTime(8, timeStep, positions, velocities, masses, numberOfBodies);
		var positions9 = CalculatePositionsAtTime(9, timeStep, positions, velocities, masses, numberOfBodies);
		var positions10 = CalculatePositionsAtTime(10, timeStep, positions, velocities, masses, numberOfBodies);
		var positions11 = CalculatePositionsAtTime(11, timeStep, positions, velocities, masses, numberOfBodies);
		var positions12 = CalculatePositionsAtTime(12, timeStep, positions, velocities, masses, numberOfBodies);
		var positions13 = CalculatePositionsAtTime(13, timeStep, positions, velocities, masses, numberOfBodies);

		var accelerations1 = new Double4[numberOfBodies];
		var accelerations2 = new Double4[numberOfBodies];
		var accelerations3 = new Double4[numberOfBodies];
		var accelerations4 = new Double4[numberOfBodies];
		var accelerations5 = new Double4[numberOfBodies];
		var accelerations6 = new Double4[numberOfBodies];
		var accelerations7 = new Double4[numberOfBodies];
		var accelerations8 = new Double4[numberOfBodies];
		var accelerations9 = new Double4[numberOfBodies];
		var accelerations10 = new Double4[numberOfBodies];
		var accelerations11 = new Double4[numberOfBodies];
		var accelerations12 = new Double4[numberOfBodies];

		for (var i = 0; i < numberOfBodies; i++)
			accelerations1[i] = CalculateTotalAcceleration(positions1, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations2[i] = CalculateTotalAcceleration(positions2, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations3[i] = CalculateTotalAcceleration(positions3, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations4[i] = CalculateTotalAcceleration(positions4, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations5[i] = CalculateTotalAcceleration(positions5, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations6[i] = CalculateTotalAcceleration(positions6, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations7[i] = CalculateTotalAcceleration(positions7, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations8[i] = CalculateTotalAcceleration(positions8, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations9[i] = CalculateTotalAcceleration(positions9, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations10[i] = CalculateTotalAcceleration(positions10, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations11[i] = CalculateTotalAcceleration(positions11, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations12[i] = CalculateTotalAcceleration(positions12, masses, numberOfBodies, i);

		var positionsArray = new Double4[numberOfBodies * 15];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i] = new Double4(0);
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 1] = positions[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 2] = positions1[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 3] = positions2[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 4] = positions3[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 5] = positions4[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 6] = positions5[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 7] = positions6[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 8] = positions7[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 9] = positions8[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 10] = positions9[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 11] = positions10[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 12] = positions11[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 13] = positions12[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 14] = positions13[i];

		var accelerationsArray = new Double4[numberOfBodies * 13];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i] = new Double4(0);
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 1] = accelerations1[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 2] = accelerations2[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 3] = accelerations3[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 4] = accelerations4[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 5] = accelerations5[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 6] = accelerations6[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 7] = accelerations7[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 8] = accelerations8[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 9] = accelerations9[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 10] = accelerations10[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 11] = accelerations11[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 12] = accelerations12[i];


		return (FlattenDoubleArray(positionsArray), FlattenDoubleArray(accelerationsArray), masses);
	}

	private static (double[] positions, double[] accelerations, double[] masses) MultistepInit(int numberOfBodies,
		double timeStep, Double4[] positions, Double4[] velocities, double[] masses, int positionArraySize, int accelerationArraySize) {

		Double4[] positionsArray = new Double4[numberOfBodies * positionArraySize];
		Double4[] accelerationsArray = new Double4[numberOfBodies * positionArraySize];

		for (var i = 0; i < numberOfBodies; i++) {
			positionsArray[i] = new Double4(0.0);
		}
		
		for (var i = 0; i < numberOfBodies; i++) {
			accelerationsArray[i] = new Double4(0.0);
		}
		
		for (var i = 1; i < positionArraySize; i++) {
			var positionsAtTime = CalculatePositionsAtTime(i - 1, timeStep, positions, velocities, masses, numberOfBodies);
			for (var j = 0; j < numberOfBodies; j++) {
				positions[i * numberOfBodies + j] = positionsAtTime[j];
			}

			if (i < accelerationArraySize) {
				var accelerationsAtTime = new Double4[numberOfBodies];
				for (var k = 0; k < numberOfBodies; k++) {
					accelerationsAtTime[k] = CalculateTotalAcceleration(positionsAtTime, masses, numberOfBodies, k);
				}

				for (var k = 0; k < numberOfBodies; k++) {
					accelerationsArray[i * numberOfBodies + k] = accelerationsAtTime[k];
				}
			}
		}
		
		return (FlattenDoubleArray(positionsArray), FlattenDoubleArray(accelerationsArray), masses);
	}

	private static double[] FlattenDoubleArray(Double4[] array) {
		var flattened = new double[array.Length * 4];
		for (var i = 0; i < array.Length; i++) {
			flattened[i * 4] = array[i].X;
			flattened[i * 4 + 1] = array[i].Y;
			flattened[i * 4 + 2] = array[i].Z;
			flattened[i * 4 + 3] = array[i].W;
		}

		return flattened;
	}

	/// <summary>
	///     time 1 = p-1, time 2 = p-1, etc.
	/// </summary>
	private static Double4[] CalculatePositionsAtTime(double time, double timeStep, Double4[] positions,
		Double4[] velocities,
		double[] masses, int numberOfBodies) {
		var positionsAtTime = new Double4[numberOfBodies];
		var velocitiesAtTime = new Double4[numberOfBodies];

		for (var i = 0; i < numberOfBodies; i++) positionsAtTime[i] = positions[i];

		for (var i = 0; i < numberOfBodies; i++) velocitiesAtTime[i] = velocities[i];

		var calcTimeStep = timeStep / 512;
		var iterations = time * timeStep / calcTimeStep;

		//Kickoff 
		for (var i = 0; i < numberOfBodies; i++)
			velocitiesAtTime[i] +=
				-calcTimeStep * 0.5 * CalculateTotalAcceleration(positionsAtTime, masses, numberOfBodies, i);

		//Leapfrog
		for (var i = 0; i < iterations; i++) {
			for (var j = 0; j < numberOfBodies; j++)
				positionsAtTime[j] += -calcTimeStep * velocitiesAtTime[j];

			for (var j = 0; j < numberOfBodies; j++)
				velocitiesAtTime[j] +=
					-calcTimeStep * CalculateTotalAcceleration(positionsAtTime, masses, numberOfBodies, j);
		}

		return positionsAtTime;
	}

	private static Double4
		CalculateTotalAcceleration(Double4[] positions, double[] masses, int numberOfBodies, int id) {
		var acceleration = new Double4(0);

		for (var i = 0; i < numberOfBodies; i++)
			if (i != id)
				acceleration += CalculateAcceleration(positions[id], positions[i], masses[i]);

		return acceleration;
	}

	private static Double4 CalculateAcceleration(Double4 position1, Double4 position2, double mass2) {
		var difference = position2 - position1;
		var distanceSquared = difference.X * difference.X + difference.Y * difference.Y + difference.Z * difference.Z;
		return mass2 / (distanceSquared * double.Sqrt(distanceSquared)) * difference;
	}
}