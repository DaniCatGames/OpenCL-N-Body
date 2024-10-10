namespace OpenCL_Barnes_Hut;

public abstract class PeriodicOrbitData {
	public static Dictionary<UniverseSetups, Double4[]> PositionLookupTable() {
		var dict = new Dictionary<UniverseSetups, Double4[]> {
			{ UniverseSetups.Infinity, [new Double4(-1.0, 0.0), new Double4(1.0, 0.0), new Double4(0.0, 0.0)] },
			{ UniverseSetups.LiLiaoUE2D, [new Double4(-1.0, 0.0), new Double4(1.0, 0.0), new Double4(0.0, 0.0)] }
		};

		return dict;
	}

	public static Dictionary<UniverseSetups, Double4[]> VelocityLookupTable() {
		var dict = new Dictionary<UniverseSetups, Double4[]> {
			{
				UniverseSetups.Infinity, [
					new Double4(0.3471168881, 0.5327249454), new Double4(0.3471168881, 0.5327249454),
					new Double4(-0.6942337762, -1.0654498908)
				]
			}, {
				UniverseSetups.LiLiaoUE2D, [
					new Double4(0.7583850283, 0.9342270211), new Double4(0.7583850283, 0.9342270211),
					new Double4(-0.7583850283, -0.9342270211)
				]
			}
		};

		return dict;
	}

	public static Dictionary<UniverseSetups, double[]> MassLookupTable() {
		var dict = new Dictionary<UniverseSetups, double[]> {
			{ UniverseSetups.Infinity, [1, 1, 1] },
			{ UniverseSetups.LiLiaoUE2D, [1, 1, 2] }
		};

		return dict;
	}
}