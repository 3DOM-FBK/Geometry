import Metashape, math

path = Metashape.app.getSaveFileName("Specify the export file path:", filter = "Text file (*.txt);;All formats (*.*)")
file = open(path, "wt")
file.write("Id\tX\tY\tZ\tvar\tcov_x\tcov_y\tcov_z\n")

chunk = Metashape.app.document.chunk
T = chunk.transform.matrix
if chunk.transform.translation and chunk.transform.rotation and chunk.transform.scale:
	T = chunk.crs.localframe(T.mulp(chunk.region.center)) * T
R = T.rotation() * T.scale()

for point in chunk.point_cloud.points:
	if not point.valid: 
		continue
	cov = point.cov
	coord = point.coord

	coord = T * coord
	cov = R * cov * R.t()
	u, s, v = cov.svd()
	var = math.sqrt(sum(s)) #variance vector length
	vect = (u.col(0) * var)
	
	var = math.sqrt(sum(s))
	
	file.write(str(point.track_id))
	file.write("\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(coord[0], coord[1], coord[2], var))
	file.write("\t{:.6f}\t{:.6f}\t{:.6f}".format(vect.x, vect.y, vect.z))
	file.write("\n")
	
file.close()
print("Script finished.")	
	
	