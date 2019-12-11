from PyQt5 import QtCore, QtWidgets, QtGui
from NodeGraphQt import NodeGraph, BaseNode, BackdropNode, setup_context_menu
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from typing import Dict, List, Tuple


class PassingGraph(NodeGraph):
    def __init__(self):
        super(PassingGraph, self).__init__()
        self.port_connected.connect(self.connected)
        self.port_disconnected.connect(self.disconnected)

    @staticmethod
    def disconnected(target_port, source_port):
        source_node, target_node = source_port.node(), target_port.node()
        try:
            source_node.disconnected(target_node.id)
            target_node.remove_upstream(source_node.id)
        except AttributeError:
            # This is not a passing node
            pass

    @staticmethod
    def connected(source_port, target_port):
        source_node, target_node = source_port.node(), target_port.node()
        try:
            source_node.connected(target_node.id, target_node.on_new_data)
            source_node.send_data()
        except AttributeError:
            # This is not a passing node
            pass


class PassingNode(BaseNode):
    # TODO: Make this work with multiple input and output nodes
    NODE_NAME = 'Passing Node'

    _down_stream: dict
    _in_data: dict
    _out_data: object

    def __init__(self):
        super(PassingNode, self).__init__()

        self._down_stream = {}
        self._in_data = {}
        self._data = None

    def process_data(self, data: List) -> object:
        """
        Define this function to process incoming data. Return the new value
        """
        return None

    def connected(self, downstream_id: str, on_data_changed):
        self._down_stream[downstream_id] = on_data_changed

    def disconnected(self, downstream_id):
        self._down_stream.pop(downstream_id, None)

    def remove_upstream(self, upstream_id):
        self._in_data.pop(upstream_id, None)
        self.set_data(self.process_data([self._in_data[id] for id in self._in_data]))

    def on_new_data(self, upstream_id: str, data: object):
        self._in_data[upstream_id] = data
        self.set_data(self.process_data([self._in_data[id] for id in self._in_data]))

    def set_data(self, data: object):
        self._data = data
        self.send_data()

    def send_data(self):
        for downstream_id in self._down_stream:
            func = self._down_stream[downstream_id]
            try:
                func(self.id, self._data)
            except Exception as e:
                print("Downstream call for", self.name(), "failed:", e)
                raise e


class MetricInputNode(PassingNode):
    __identifier__ = "com.vidproc"
    NODE_NAME = "Metric Input Node"

    def __init__(self):
        super(MetricInputNode, self).__init__()

    def set_metric(self, name: str, data: np.ndarray):
        self.set_data((name, data))
        self.add_output(name)


class NormalizeNode(PassingNode):
    __identifier__ = 'com.vidproc'
    NODE_NAME = "Normalize Data Node"

    def __init__(self):
        super(NormalizeNode, self).__init__()
        self.add_input("Data", multi_input=True)
        self.add_output("Normalized Data", multi_output=True)

    def process_data(self, inp: List[Tuple[str, np.ndarray]]) -> List[Tuple[str, np.ndarray]]:
        inps = []
        for datum in inp:
            if isinstance(datum, list):
                inps.extend(datum)
            else:
                inps.append(datum)
        out = []
        for datum in inps:
            if datum is None:
                continue
            name, data = datum
            data = data.copy()
            shift = data.min()
            factor = data.max()-shift
            out.append((name, (data-shift)/factor))
        return out[0] if len(out) == 1 else out


class SmoothNode(PassingNode):
    __identifier__ = "com.vidproc"
    NODE_NAME = "Smooth Data Node"

    def __init__(self):
        super(SmoothNode, self).__init__()
        self.add_input("Data", multi_input=True)
        self.add_output("Smoothed Data", multi_output=True)

    def process_data(self, inp: List) -> object:
        inps = []
        for datum in inp:
            if isinstance(datum, list):
                inps.extend(datum)
            else:
                inps.append(datum)
        out = []
        for datum in inps:
            name, data = datum
            data = data.copy()
            if len(data) >= 5:
                window = min(len(data), 51)
                if window % 2 == 0:
                    window -= 1
                smoothed = savgol_filter(data, window, 3)
            else:
                smoothed = data
            out.append((name, smoothed))
        return out[0] if len(out) == 1 else out

class GrapherNode(PassingNode):
    __identifier__ = "com.vidproc"
    NODE_NAME = "Graph Data"

    def __init__(self):
        super(GrapherNode, self).__init__()
        self.add_input("Data", multi_input=True)

    def callback(self, data):
        print("Callback not set for:", self.name())

    def set_callback(self, callback):
        self.callback = callback

    def process_data(self, data: List) -> List[Tuple[str, np.ndarray]]:
        inps = []
        for datum in data:
            if isinstance(datum, list):
                inps.extend(datum)
            else:
                inps.append(datum)
        self.callback(inps)

class MetricFlow(PassingGraph):
    _metrics: Dict[str, np.ndarray]

    _input_nodes: Dict[str, MetricInputNode]

    def __init__(self, parent=None):
        super(MetricFlow, self).__init__()
        setup_context_menu(self)

        self._input_nodes = {}

        reg_nodes = [
            MetricInputNode,
            NormalizeNode,
            SmoothNode,
            GrapherNode
        ]
        for node in reg_nodes:
            self.register_node(node)

        self.node_created.connect(self.on_new_node)

    def callback(self, data):
        print("No grapher callback defined")

    def on_new_node(self, node):
        if isinstance(node, GrapherNode):
            node.set_callback(self.on_graph)

    def set_metric_data(self, metric_data: pd.DataFrame):
        self._metrics = metric_data.to_dict()
        self._metrics.pop("Frame_number", None)
        for i, metric_name in enumerate(self._metrics):
            # Create a MetricInputNode
            data = self._metrics[metric_name]
            data = np.array([data[frame] for frame in data])
            my_node: MetricInputNode = self.create_node('com.vidproc.MetricInputNode',
                                        name=metric_name,
                                        color='#0a1e20',
                                        text_color='#feab20',
                                        pos=[0, i*100+10])
            my_node.set_metric(metric_name, data)
            self._input_nodes[metric_name] = my_node
        self.fit_to_selection()
        self.viewer()._set_viewer_pan(-600, 0)
        menu = self.context_menu()
        file_menu = menu.get_menu("&File")
        file_menu.qmenu.menuAction().setVisible(False)
        commands = menu.all_commands()
        to_remove = ["Copy", "Paste"]
        for command in commands:
            if command.name() in to_remove:
                command.qaction.setVisible(False)
            elif command.name() == "Delete":
                print(QtGui.QKeySequence(QtGui.QKeySequence.Delete).toString())
                command.qaction.setShortcuts([16777219, QtGui.QKeySequence.Delete])
        return

    def on_graph(self, data: List[Tuple[str, np.ndarray]]):
        self.callback(data)

    def set_graph_callback(self, callback):
        self.callback = callback

    def show(self):
        self.viewer().show()

    def hide(self):
        self.viewer().hide()
