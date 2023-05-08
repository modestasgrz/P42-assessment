from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import networkx as netx
import numpy as np

from neurons import ActionNeuron, InnerNeuron, SensorNeuron

matplotlib.use("TkAgg")


class Brain:  # Directional graph
    def __init__(self, genome, config):
        self.connections = self.read_genome(
            genome,
            config.num_sensory_neurons,
            config.num_inner_neurons,
            config.num_action_neurons,
        )
        self.connections = self.remove_redundant_connections(self.connections)
        self.brain = self.make_brain_dictionary(self.connections)
        self.sense_neurons_mapping = config.sense_neurons_mapping
        self.brain_graph = None

    def __str__(self):
        str_brain = "{\n"
        for key, value in self.brain.items():
            str_connections = "{ \n"
            for neuron, weight in value.items():
                str_connections += " " * 13 + str(neuron) + " :  " + str(weight) + ", \n"
            str_connections += "          }"
            str_brain += f"   {str(key)} : {str_connections}" + "\n"
        str_brain += "}"
        return str_brain

    # def forward(self, x): here initial x - dictionary of sense input values mapping
    def __call__(self, sense_input):  # functionality - debug this
        # This is the where calculated values are accumulated
        activated_sinks = self.sensory_neurons_activation(sense_input)
        activated_sinks = self.inner_neurons_activation(activated_sinks)
        output_activations = self.action_neurons_activation(activated_sinks)

        # Find index of the action neuron having the largest activation
        _, an_index = np.argmax(
            np.array([(str(sink), value) for (sink, value) in output_activations]),
            axis=0,
        )

        # Formulate output
        action_neuron, activation = output_activations[an_index]  # pylint:disable = unused-variable

        return action_neuron.activate()  # Provides number of in-game action - neuron's type id

    def sensory_neurons_activation(self, sense_input: Dict):
        activated_sinks = []
        x = 0.0
        # Sensory Neurons activation
        # Find Sensory Neurons and calculate activations,
        # Then append them to the activated_sinks list
        for source, sinks in self.brain.items():
            if "SN" in str(source):
                for sink, weight in sinks.items():
                    activation_input_type = self.sense_neurons_mapping.get(str(source), "")
                    x = sense_input.get(activation_input_type, 0.0)
                    x = source(x, weight)
                    activated_sinks.append((sink, x))

        return activated_sinks

    def inner_neurons_activation(self, activated_sinks):
        # Inner Neurons activation
        # Find some Inner Neuron
        for new_source, _ in activated_sinks:
            if "IN" in str(new_source):
                current_in = str(new_source)
                # For every activation assigned to the chosen Inner Neuron collect input
                in_input = [x for source, x in activated_sinks if str(source) == current_in]

                # Remove registered activation from the list
                # - filter list, so it could not interfere with later runs
                activated_sinks = [
                    (node, activation) for (node, activation) in activated_sinks if str(node) != current_in
                ]
                # Now determine, to which new sinks assign this activation, if input exists
                if in_input:
                    for source, sinks in self.brain.items():
                        activated_sinks = [
                            (sink, source(in_input, weight))
                            for sink, weight in sinks.items()
                            if current_in == str(source)
                        ]

        return activated_sinks

    def action_neurons_activation(self, activated_sinks):
        # Action Neurons activation
        # First declare output activations list
        output_activations = []
        # Find some Action Neuron - begining analogous to IN activation
        for new_source, _ in activated_sinks:
            if "AN" in str(new_source):
                current_an = str(new_source)
                # For every activation assigned to the chosen Action Neuron collect input
                an_input = [x for source, x in activated_sinks if str(source) == current_an]
                # Remove registered activation from the list
                # - filter list, so it could not interfere with later runs
                activated_sinks = [
                    (node, activation) for (node, activation) in activated_sinks if str(node) != current_an
                ]
                # Now calculate activations for every action neuron and put them in a list, if inputs exist
                if an_input:
                    x = new_source(an_input)
                    output_activations.append((new_source, x))

        return output_activations

    def display_brain(self):
        self.brain_graph = netx.DiGraph()
        str_connections = [
            (str(source_neuron), str(sink_neuron), round(weight, 2))
            for source_neuron, sink_neuron, weight in self.connections
        ]
        self.brain_graph.add_weighted_edges_from(str_connections)
        colors = {}
        for node in self.brain_graph.nodes():
            if "SN" in node:  # Sensory Neuron
                colors[node] = "blue"
            elif "IN" in node:  # Inner Neuron
                colors[node] = "grey"
            else:
                colors[node] = "red"
        weight_labels = netx.get_edge_attributes(self.brain_graph, "weight")
        node_labels = {node: node for node in self.brain_graph.nodes()}
        # Spring layout - appealing to the eye layout of a graph, required for draw function
        pos = netx.spring_layout(self.brain_graph)
        # netx.draw(self.brain_graph, with_labels=True)
        netx.draw_networkx_nodes(
            self.brain_graph,
            pos,
            node_color=[colors.get(node, "gray") for node in self.brain_graph.nodes()],
            alpha=0.3,
        )
        netx.draw_networkx_edges(self.brain_graph, pos)
        netx.draw_networkx_edge_labels(self.brain_graph, pos, edge_labels=weight_labels)
        netx.draw_networkx_labels(self.brain_graph, pos, labels=node_labels, font_size=10)
        plt.show()

    def make_brain_dictionary(self, connections):  # make brain graph
        brain = {}
        for connection in connections:
            source_neuron, sink_neuron, weight = connection
            appended = False
            for existing_neuron in brain:
                if source_neuron.type_id == existing_neuron.type_id and str(source_neuron) == str(existing_neuron):
                    # Appends sink neuron to existing source neuron key

                    # Every neuron at init stage is assigned with a different reference,
                    # but neurons with the same type and id should match in reference
                    # as they are the same one neuron
                    source_neuron = existing_neuron
                    brain[source_neuron].append((sink_neuron, weight))
                    appended = True
                    break
            if not appended:
                # Adds new source neuron and corresponding sink neuron to it
                brain[source_neuron] = [(sink_neuron, weight)]
        for key in brain:
            # Switching lists in values instead of dictionaries
            brain[key] = dict(brain[key])

        return brain

    def remove_redundant_connections(self, connections):
        # Make neuron list with num_inputs and num_outputs
        neuron_list = []  # (neuron, num_inputs, num_outputs)
        # print(np.array([(str(source), str(sink), weight) for (source, sink, weight) in connections]))
        # print("")
        for source, sink, weight in connections:
            if str(source) == str(sink):
                # Illegal connection - remove imediately
                connections.remove((source, sink, weight))
            source_updated = False
            sink_updated = False
            for i, item in enumerate(neuron_list):
                neuron, num_inputs, num_outputs = item
                if str(source) == str(neuron):
                    neuron_list[i] = (neuron, (num_inputs + 1), num_outputs)
                    source_updated = True
                if str(sink) == str(neuron):
                    neuron_list[i] = (neuron, num_inputs, (num_outputs + 1))
                    sink_updated = True
            if not source_updated:
                neuron_list.append((source, 1, 0))
            if not sink_updated:
                neuron_list.append((sink, 0, 1))

        # Find redundant neurons
        # Redundant neurons - INs with 0 inputs OR 0 outputs are redundant
        # - connections with them will be removed - names of the neurons
        redundant_neurons = [
            str(neuron)
            for (neuron, num_inputs, num_outputs) in neuron_list
            if "IN" in str(neuron) and (num_inputs == 0 or num_outputs == 0)
        ]

        # Remove them from connections - filter connections
        connections = [
            (source, sink, weight)
            for (source, sink, weight) in connections
            if (not str(source) in redundant_neurons) and (not str(sink) in redundant_neurons)
        ]

        # print(np.array([(str(source), str(sink), weight) for (source, sink, weight) in connections]))
        # print("")
        # print(np.array([neuron for neuron in redundant_neurons]))
        return connections

    def read_gene(self, gene, num_sensory_neurons, num_inner_neurons, num_action_neurons):
        # Hex string to Bin string
        gene = format(int(gene, 16), "032b")

        # Parse encoded parameters from bits
        # Source Type: 0 - sensory neuron; 1 - inner neuron
        source_type = gene[0]
        source_type = int(source_type, 2)

        # Source ID: type id of the source neuron (without modulus applied)
        source_id = gene[1:8]
        source_id = int(source_id, 2)

        # Sink Type: 0 - action neuron; 1 - inner neuron
        sink_type = gene[8]
        sink_type = int(sink_type, 2)
        # NEW RULE: Inner Neurons cannot connect to themselves or each other
        if source_type == 1:
            sink_type = 0

        # Sink ID: type id of the sink neuron (without modulus applied)
        sink_id = gene[9:16]
        sink_id = int(sink_id, 2)

        # Weight: weight of the connection <- [-5 ... 5]
        weight = gene[16:]
        weight = (((int(weight, 2) / int("ffff", 16)) - 0.5) * 2) * 5  # [-5 .. 5]

        if source_type == 1:  # Inner neuron
            source_id = source_id % num_inner_neurons
            source_neuron = InnerNeuron(type_id=source_id)
        elif source_type == 0:  # Sensor neuron
            source_id = source_id % num_sensory_neurons
            source_neuron = SensorNeuron(type_id=source_id)

        if sink_type == 1:  # Inner neuron
            sink_id = sink_id % num_inner_neurons
            sink_neuron = InnerNeuron(type_id=sink_id)
        elif sink_type == 0:  # Action neuron
            sink_id = sink_id % num_action_neurons
            sink_neuron = ActionNeuron(type_id=sink_id)

        return source_neuron, sink_neuron, weight

    def read_genome(self, genome, num_sensory_neurons, num_inner_neurons, num_action_neurons):
        return [self.read_gene(gene, num_sensory_neurons, num_inner_neurons, num_action_neurons) for gene in genome]
