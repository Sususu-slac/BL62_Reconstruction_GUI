"""One GPU block, executed by the isolated Python/GI runtime without CLI forks."""
from pathlib import Path
import json, os, sys
import gi
gi.require_version('Ufo','1.0')
from gi.repository import GObject, Ufo

def typed(values,kind=GObject.TYPE_FLOAT):
    array=GObject.ValueArray.new(len(values))
    for value in values:
        item=GObject.Value();item.init(kind)
        if kind==GObject.TYPE_INT:item.set_int(int(value))
        else:item.set_float(float(value))
        array.append(item)
    return array

def main():
    info=json.loads(Path(sys.argv[1]).read_text());job=info['job'];block=info['block'];work=Path(info['work'])
    if os.environ.get('UFO_DEVICE_TYPE')!='gpu':raise RuntimeError('GPU device type is required; CPU fallback is disabled')
    manager=Ufo.PluginManager();graph=Ufo.TaskGraph();scheduler=Ufo.FixedScheduler()
    nodes=scheduler.get_resources().get_gpu_nodes()
    if not nodes:raise RuntimeError('No OpenCL GPU is available')
    source=manager.get_task('read');source.set_properties(path=str(work/'projections.tiff'),number=job['nproj'],height=job['height'])
    back=manager.get_task('lamino-backproject');back.set_proc_node(nodes[0])
    back.props.num_projections=job['nproj'];back.props.overall_angle=job['overall_angle']
    back.props.lamino_angle=block['lamino_angle'];back.props.roll_angle=0.
    back.props.x_region=typed(job['x_region'],GObject.TYPE_INT);back.props.y_region=typed(job['y_region'],GObject.TYPE_INT)
    back.props.center=typed(block['center']);back.props.region=typed(block['region']);back.props.z=block['z']
    back.props.parameter=block['parameter'];back.props.addressing_mode='clamp'
    slicer=manager.get_task('slice');slicer.set_proc_node(nodes[0])
    writer=manager.get_task('write');writer.props.filename=str(work/'block.tiff')
    writer.props.append=False
    if hasattr(writer.props,'bytes_per_file'):writer.props.bytes_per_file=4*1024**3
    if hasattr(writer.props,'tiff_bigtiff'):writer.props.tiff_bigtiff=True
    graph.connect_nodes(source,back);graph.connect_nodes(back,slicer);graph.connect_nodes(slicer,writer)
    scheduler.run(graph)
    print('TOFU_EXECUTION_SECONDS '+str(scheduler.props.time),flush=True)

if __name__=='__main__':main()
