Overview:
 - Read in image
 - perspective_warp the image
 - split channels and identify object/fields
 - combine fields after object filter
 - add edge blur to goal image
 - Create risk image
 - take image and aff gridify filter
 - convert gridified image cells into start, finish and goal States
 - parse through LTL Formula
 - for each goal create a reward image of where it is



 - parse LTL file

 - Convert LTL automata into reward automata
 - show current






 Figure out which "finish" cordiante matches up with which node in the LTL Formula
 we knoe that the finish cordiante satifies one of the transitions
 for each transition figure out which transition satifies
 make the "finish" coord tthe start and restart the process
