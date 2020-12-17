cmd="python build_graph.py -dataset moocen -no_course_dependency -no_user_act"
echo $cmd & $cmd
cmd="python postprocess.py -dataset moocen -model gcn -info main_v"
echo $cmd & $cmd

cmd="python build_graph.py -dataset mooczh"
echo $cmd & $cmd
cmd="python postprocess.py -dataset mooczh -model gcn -info main_cvs"
echo $cmd & $cmd
