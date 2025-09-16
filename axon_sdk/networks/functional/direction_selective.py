from axon_sdk.primitives import SpikingNetworkModule, DataEncoder
from typing import Optional

class DirectionSelectiveDoublet(SpikingNetworkModule):
    """
    A->B direction-selective unit using the doublet method adapted to STICK from: 
    G. Haessig, A. Cassidy, R. Alvarez, R. Benosman and G. Orchard, "Spiking Optical Flow for Event-Based Sensors Using IBM's TrueNorth Neurosynaptic System," 
    in IEEE Transactions on Biomedical Circuits and Systems, vol. 12, no. 4, pp. 860-870, Aug. 2018, doi: 10.1109/TBCAS.2018.2834558
    """

    def __init__(
        self,
        encoder: DataEncoder,
        Vt: float = 10.0,
        tm: float = 100.0,
        tf: float = 20.0,
        Ts_A_to_OA= 1.0,      # A -> O_A (timestamp A)
        Ts_B_to_OB = 1.0,      # B -> O_B (candidate B spike) 
        Ts_B_to_G = 0.2,      # B -> G   (arm guard)
        Ts_G_to_OB= 0.1,      # G -> O_B (inhibit O_B)
        Ts_A_to_G= 0.2,      # A -> G   (disarm guard)
        Ts_OA_to_C = 1.0,      # O_A -> C (first spike)
        Ts_OB_to_C= 1.0,      # O_B -> C (second spike)
        k_inhib: float = 2.0,
        module_name: Optional[str] = None,
    ) -> None:
        super().__init__(module_name)
        self.encoder = encoder
        we = Vt
        wi = -k_inhib * Vt

        self.A  = self.add_neuron(Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="A_in")
        self.B  = self.add_neuron(Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="B_in")
        self.OA = self.add_neuron(Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="O_A")
        self.OB = self.add_neuron(Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="O_B")
        self.G  = self.add_neuron(Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="Guard")
        self.C  = self.add_neuron(Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="Collector")

        # Timestamp A and candidate B
        self.connect_neurons(self.A, self.OA, "V", we, Ts_A_to_OA)
        self.connect_neurons(self.B, self.OB, "V", we, Ts_B_to_OB)     

        # Guard logic: B arms guard quickly; guard inhibits O_B even faster.
        self.connect_neurons(self.B, self.G,  "V", we, Ts_B_to_G)    
        self.connect_neurons(self.G, self.OB, "V", wi, Ts_G_to_OB)     # inhibition lands BEFORE B excitation

        # A disarms guard quickly so later B is permitted (preferred direction)
        self.connect_neurons(self.A, self.G, "V", wi, Ts_A_to_G)

        # Collector: convert O_A and O_B timestamps to a single-neuron doublet on C
        self.connect_neurons(self.OA, self.C, "V", we, Ts_OA_to_C)
        self.connect_neurons(self.OB, self.C, "V", we, Ts_OB_to_C)

    @property
    def output(self):
        return self.C

    @property
    def inputs(self):
        return self.A, self.B


if __name__ == "__main__":
    from axon_sdk import Simulator
    enc = DataEncoder()

    # Preferred Direction A->B
    ds = DirectionSelectiveDoublet(enc)
    sim1 = Simulator(ds, enc)
    A, B = ds.inputs
    tA, tB = 10.0, 26.0
    sim1.apply_input_spike(A, t=tA)
    sim1.apply_input_spike(B, t=tB)
    sim1.simulate(simulation_time=80.0)
    print("=== Preferred direction (A -> B) ===")
    print(f"O_A spikes: {sim1.spike_log[ds.OA.uid]}")
    print(f"O_B spikes: {sim1.spike_log[ds.OB.uid]}")
    C_spikes = sim1.spike_log[ds.output.uid]
    print(f"C (collector) spikes: {C_spikes}")
    if len(C_spikes) == 2:
        print(f"Measured ISI on C: {C_spikes[1]-C_spikes[0]:.3f}")
        print(f"Ground truth (tB - tA): {tB - tA:.3f}")

    # Null B->A (O_B must be suppressed; C must not doublet)
    ds2 = DirectionSelectiveDoublet(enc)
    sim2 = Simulator(ds2, enc)
    A2, B2 = ds2.inputs
    tB2, tA2 = 10.0, 26.0
    sim2.apply_input_spike(B2, t=tB2)
    sim2.apply_input_spike(A2, t=tA2)
    sim2.simulate(simulation_time=80.0)
    print("\n=== Null direction (B -> A) ===")
    print(f"O_A spikes: {sim2.spike_log[ds2.OA.uid]}")
    print(f"O_B spikes: {sim2.spike_log[ds2.OB.uid]}   (should be [])")
    print(f"C (collector) spikes: {sim2.spike_log[ds2.output.uid]}   (should be 0 or 1)")

    # Fast A->B (refractory check)
    ds3 = DirectionSelectiveDoublet(enc)
    sim3 = Simulator(ds3, enc)
    A3, B3 = ds3.inputs
    tA3, tB3 = 10.0, 12.5
    sim3.apply_input_spike(A3, t=tA3)
    sim3.apply_input_spike(B3, t=tB3)
    sim3.simulate(simulation_time=50.0)
    print("\nFast A -> B (refractory check)")
    print(f"C (collector) spikes: {sim3.spike_log[ds3.output.uid]}")
