from abc import ABCMeta, abstractmethod
import logging
from collections import namedtuple
import numpy as np
import requests
import json 

LOG = logging.getLogger(__name__)

class DataInOutForGeometry(namedtuple('DataInOutForGeometry', ['geometry_type', 'data_in', 'data_out',
                                                               'time_axis', 'cutoff_ind',
                                                               'wells_in', 'horizons_in', 'planars_in',
                                                               'planars_out',
                                                               'geometry'])):
    __slots__ = ()


class TracesComputationsEnvironment(namedtuple('TracesComputationsEnvironment',
                                               'i j x y z_axis n_points n_dim n_in n_out data_in hor_values planars_values other')):
    __slots__ = ()

class NameAndFlavour(namedtuple('NameAndFlavour', 'name flavour')):
    """Tuple describing name and flavour (qn, ql, rgb) of data.
    """
    __slots__ = ()

class InputDataItem(namedtuple("InputDataItem", "file_name name")):
    ""
    __slots__ = ()

class Geometry:
    "General geometry descriptor"
    
    @property
    def type(self):
        pass
    @property
    def input_data_items(self):
        pass

class di_objects:
    "Mock-up interface"
    @staticmethod
    def job_descriptor(runargs):
        pass

    class messager:
        @classmethod
        def Messager(cls):
            pass

    class trace_data:
        @staticmethod
        def SeisCubeReader():
            pass
        def SeisLineReader():
            pass
        def SeisCubeWriter():
            pass
        def SeisLineWriter():
            pass

class DiJob:
    def __init__(self, job_id: int) -> None:
        self.job_id = job_id
        self.server_url = ""
        self._description = None

    @property
    def description(self):
        if self._description is None:
            with requests.get(f"{self.server_url}/jobs/info/{self.job_id}/") as resp:
                if resp.status_code != 200:
                    LOG.error(f"unable to get job description from {self.server_url} for {self.job_id}")
                    raise RuntimeError(f"unable to get job description from {self.server_url} for {self.job_id}")
                resp_j = json.loads(resp.content)
                self._description = resp_j["job_description"]
        return self._description




class DIJob(metaclass=ABCMeta):
    def __init__(self, runargs_filename, job_id, session_id=None, db_server_url=None):
        self.runargs = di_objects.job_descriptor.parse_job_descriptor(runargs_filename)
        self.job_id = job_id
        self.session_id = session_id
        self.db_server_url = db_server_url
        self.data_names_flavours = []  # list of NameAndFlavour
        self.planar_names_flavours = []
        self.job_datasets = []
        self.messager = di_objects.messager.Messager()
        self.total_trace_number = 0
        self.processed_traces = 0
        self.additional_env_parameters = {}

    def set_data_names(self, data_names):
        self.data_names_flavours = [NameAndFlavour(n, 'qn') for n in data_names]

    def set_data_names_flavours(self, data_names_favours, default_flavour='qn'):
        for n in data_names_favours:
            if type(n) == tuple:
                self.data_names_flavours.append(NameAndFlavour(n[0], n[1]))
            elif type(n) == str:
                self.data_names_flavours.append(NameAndFlavour(n, default_flavour))
        return self

    def set_planar_names(self, planar_names):
        self.planar_names_flavours = [NameAndFlavour(n, 'qn') for n in planar_names]

    @property
    def geometries(self):
        return self.runargs.geometries

    @property
    def parameters_dict(self):
        return self.runargs.parameters

    def int_parameter(self, pname):
        return int(self.parameters_dict[pname])

    def float_parameter(self, pname):
        return float(self.parameters_dict[pname])

    def str_parameter(self, pname):
        return self.parameters_dict[pname]

    def _create_datasets(self):
        LOG.debug('creating datasets...')
        for g in self.runargs.geometries:
            data_in = self._create_data_sources(g)
            data_geometry, time_axis, ij = self._calculate_intersecting_geometry_and_time(data_in)
            data_out = self._create_resulting_data(g, data_geometry, time_axis)
            self._close_data(data_in)
            self._close_data(data_out)
            horizons_in = self._create_horizons_in(g)
            planars_in = self._create_planars_in(g)
            planars_out = self._create_planars_out(g, data_geometry) # Planars are created in the closed state
            self.job_datasets.append(DataInOutForGeometry(geometry_type=g.type, data_in=data_in,
                                                          data_out=data_out, time_axis=time_axis, cutoff_ind=ij,
                                                          wells_in=[], horizons_in=horizons_in, planars_in=planars_in,
                                                          planars_out=planars_out,
                                                          geometry=data_geometry))
            LOG.debug('Added dataset: %s', self.job_datasets[-1])

    def _setup_job(self):
        LOG.debug('Setting job')
        LOG.debug('Job parameters: %s', self.parameters_dict)
        self._create_datasets()
        self._calculate_total_trace_number()
        self.processed_traces = 0

    def _finish_computations(self):
        LOG.debug('Finishing computations')

    def geometry_iter(self):
        """
        Iterates over all the geometries enumerated in the runargs file.
        :return: Instance of DataInOutForGeometry, all data sources and data outputs are open
        """
        for g in self.job_datasets:
            self._reopen_data(g.data_in)
            self._reopen_data(g.data_out)
            self._reopen_data(g.planars_out)
            LOG.debug('reopening data %s - %s', g.data_in, g.data_out)
            yield g
            LOG.debug('closing data in %s', g)
            self._close_data(g.data_in)
            self._close_data(g.data_out)
            self._close_data(g.planars_out)

    def _create_data_sources(self, geom: Geometry):
        """
        Method creates data sources
        :type geom: Geometry
        :returns List of input objects corresponding to trace data
        """
        if geom.type == 'cube':
            create_class = di_objects.trace_data.SeisCubeReader
        elif geom.type == 'line':
            create_class = di_objects.trace_data.SeisLineReader
        tmp = [InputDataItem(os.path.join(PROJECTS_DIR, d.path), d.name) for d in geom.input_data_items]
        return [create_class(p.path, object_name=p.name) for p in tmp]

    def _create_horizons_in(self, geom: Geometry):
        """
        Create objects representing 2D or 3D horizons corresponding to the given geomentry
        """
        if geom.type == 'cube':
            create_class = di_objects.planars.Horizon3D
        elif geom.type == 'line':
            create_class = di_objects.planars.Horizon2D
        tmp = [InputDataItem(os.path.join(PROJECTS_DIR, d.path), d.name) for d in geom.horizons]
        return [create_class(p.path, object_name=p.name) for p in tmp]
    
    def _create_planars_in(self, geom: Geometry):
        """
        Create objects representing 2D or 3D horizons corresponding to the given geomentry
        """
        if geom.type == 'cube':
            create_class = di_objects.planars.Planar3D
        elif geom.type == 'line':
            create_class = di_objects.planars.Planar2D
        tmp = [InputDataItem(os.path.join(PROJECTS_DIR, d.path), d.name) for d in geom.planars]
        return [create_class(p.path, object_name=p.name) for p in tmp]
        
    def _create_planars_out(self, geom: Geometry, data_geometry):
        """
        Create the output planars
        """
        LOG.info("In _create_resulting_data: %s", geom)
        LOG.info("In _create_resulting_data data geometry: %s", data_geometry)
        if geom.type == 'cube':
            create_class = di_objects.planars.PlanarWriter3D
            geometry_legacy = Horizon3DGeometry(data_geometry[3], data_geometry[4], data_geometry[0][:2], data_geometry[1], data_geometry[2])
        elif geom.type == 'line':
            create_class = di_objects.planars.PlanarWriter2D
            geometry_legacy = data_geometry
        else:
            raise RuntimeError('Unsupported geometry type: %s' % geom.type )
        # tmp = [InputDataItem(self._generate_out_file_name(geom.output_location, n.name, flavour=n.flavour, data_type='sc'), n.name) 
        #         for n in self.data_names_flavours]
        planars_out = []
        for nf in self.planar_names_flavours:
            planar_name = nf.name
            planar_file_name = self._generate_out_file_name(geom.output_location, planar_name, flavour='qn', data_type='sc')
            tmp = InputDataItem(planar_file_name, planar_name)
            LOG.info("Planars out: %s", tmp)
            # return [create_class(object_name=p.name, file_name=p.path, geom=data_geometry, time_axis=time_axis) for p in tmp]
            planar = create_class(object_name=planar_name, file_name=planar_file_name, geometry=geometry_legacy)
            planar.close()
            planars_out.append(planar)
        return planars_out

    def _calculate_intersecting_geometry_and_time(self, data_in):
        geom = data_in[0].geometry
        time_axis, ij = di_objects.trace_data.intersect_axes(*[d.z_axis for d in data_in])
        # make sure the input geometries data coincide:
        for d in data_in[1:]:
            assert d.is_geometry_same(data_in[0]), 'Incompatible geometries!'
        return geom, time_axis, ij

    def _create_resulting_data(self, geom: Geometry, data_geometry, time_axis):
        """
        Creates open resulting data objects (writers)
        :type geom: Geometry
        """
        if geom.type == 'cube':
            create_class = di_objects.trace_data.SeisCubeWriter
        elif geom.type == 'line':
            create_class = di_objects.trace_data.SeisLineWriter
        tmp = [InputDataItem(self._generate_out_file_name(geom.output_location, n.name, flavour=n.flavour), n.name) 
                for n in self.data_names_flavours]
        return [create_class(object_name=p.name, file_name=p.path, geom=data_geometry, time_axis=time_axis) for p in tmp]

    def _setup_processing_geometry(self, g: DataInOutForGeometry):
        LOG.debug('Setting up %s', g)

    def _finish_processing_geometry(self, g: DataInOutForGeometry):
        LOG.debug('Finishing %s', g)

    def process_single_geometry(self, g: DataInOutForGeometry):
        def n_dim(g_type):
            if g_type == 'line':
                return 2
            elif g_type == 'cube':
                return 3
            else:
                raise RuntimeError('Invalid dimensions of geometry: ' + g_type)

        LOG.debug('Starting processing %s', g)
        data = np.zeros((len(g.data_in), g.time_axis.n_points))
        self._setup_processing_geometry(g)
        traces_generators = [d.next_trace() for d in g.data_in]
        while True:
            try:
                tr_in = [next(it) for it in traces_generators]
                tr_in_data = list(zip([t.data for t in tr_in], g.cutoff_ind))
                tr = tr_in[0]   # Here we assume that input geometries coincide, so we can take
                                # trace geometrical parameters from the 1st dataset
                env = TracesComputationsEnvironment(i=tr.i, j=tr.j, x=tr.x, y=tr.y,
                                                    z_axis=g.time_axis, n_points=g.time_axis.n_points,
                                                    n_dim=n_dim(g.geometry_type),
                                                    n_in=len(g.data_in), n_out=len(g.data_out), data_in=g.data_in,
                                                    hor_values=[h.at_xy(tr.x, tr.y) for h in g.horizons_in],
                                                    planars_values=[p.at_xy(tr.x, tr.y) for p in g.planars_in],
                                                    other=self.additional_env_parameters)
                for i in range(env.n_in):
                    i_start = tr_in_data[i][1][0]
                    i_end = tr_in_data[i][1][1]
                    data[i] = tr_in_data[i][0][i_start:i_end]
                res_block = self.process_block_of_traces(data, env)
                for i in range(len(g.data_out)):
                    tr_out = Trace(i=tr.i, j=tr.j, x=tr.x, y=tr.y, z0=tr.z0, dz=tr.dz, data=res_block[i])
                    g.data_out[i].put_trace(tr_out)
                self.processed_traces += 1
                self.messager.setGauge(self.processed_traces, self.total_trace_number)
            except StopIteration:
                break
        self._finish_processing_geometry(g)

    @abstractmethod
    def process_block_of_traces(self, data_in: np.ndarray, env: TracesComputationsEnvironment):
        """
        Process block of traces having the shape (n_data_in, n_points) into the output block
         of shape (n_data_out, n_points), where n_data_in and n_data_out are numbers of input and
         output data, respectively.
         This function should be redefined by the extending class.

         data_in: Array of shape (n_data_in, n_points)
         env: computing environment, env.data may contain additional parameters required for performing  computations
         over the input block of traces.
         :returns np.array of the shape (n_data_out, n_points)
        """
        res = np.zeros((env.n_out, env.n_points))
        assert len(data_in.shape) == 2
        res[:] = data_in[0]  # Using broadcast to fill the whole array from the first input dataset
        return res

    def _generate_out_file_name(self, output_location, name, data_type='tr', flavour='qn'):
        nm = "{name}_{data_type}_{flavour}__{job_id}.dx".format(name=name, data_type=data_type, 
                        flavour=flavour, job_id=self.job_id)
        return os.path.join(PROJECTS_DIR, output_location, nm)

    @staticmethod
    def _close_data(data_list):
        for d in data_list:
            d.close()

    @staticmethod
    def _reopen_data(data_list):
        for d in data_list:
            d.reopen()

    def _calculate_total_trace_number(self):
        self.total_trace_number = 0
        for g in self.job_datasets:
            if len(g.data_out) > 0:
                self.total_trace_number += g.data_out[0].trace_count
            elif len(g.data_in) > 0:
                self.total_trace_number += g.data_in[0].trace_count
        LOG.info('Total trace number %d', self.total_trace_number)

    def run(self):
        self._setup_job()
        for g in self.geometry_iter():
            self.process_single_geometry(g)
        self._finish_computations()

    @staticmethod
    def run_job(job_class, program_description: str):
        import argparse
        import time

        global LOG

        parser = argparse.ArgumentParser(description=program_description)
        parser.add_argument('runargs', help='Job description file')
        parser.add_argument('job_id', help='Job id to use in data names')
        parser.add_argument('session_id', default=None, nargs='?', help='(Optional) Session ID')
        parser.add_argument('db_url', default=None, nargs='?', help='(Optional) DB Server URL')

        parser.add_argument('-e', '--emulate', help='Emulation level, the default value is 0 (no emulation)',
                            default=0, type=int, choices=range(2))
        parser.add_argument('-l', '--loglevel', help='Log level', default='INFO',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        args = parser.parse_args()
        LOG.debug('Parsed arguments %s', args)
        logging.basicConfig(level=args.loglevel)
        LOG.info("******* ReviewJob parameters:  %s %s %s %s", args.runargs, args.job_id, args.session_id,
                    args.db_url)
        start_time = time.time()
        j1 = job_class(args.runargs, job_id=args.job_id)
        LOG.info('Parameters %s', j1.parameters_dict)
        j1.run()
        LOG.info('Total time elapsed on the job: %.2fs', time.time() - start_time)

