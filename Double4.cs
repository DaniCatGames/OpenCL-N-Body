namespace OpenCL_Barnes_Hut;

internal struct Double4 {
	public readonly double X;
	public readonly double Y;
	public readonly double Z;
	public readonly double W;

	public Double4(double x, double y, double z) {
		X = x;
		Y = y;
		Z = z;
		W = 0.0;
	}

	public Double4(double x) {
		X = x;
		Y = x;
		Z = x;
		W = 0.0;
	}
}