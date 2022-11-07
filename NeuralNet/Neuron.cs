using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNet
{
    /// <summary>
    /// See 'Net' class for details. Note that the bias is implemented as a weight from a neuron with a constant
    /// activation of '1'. A list of inputs to the neuron are also held in this class. If you delete a neuron you should
    /// also iterate the neurons in the next level and delete the connection from the deleted neuron to that neuron in
    /// the next layer--otherwise the deleted neuron's last activation value will always be used to drive the next
    /// neuron.
    /// </summary>
    public class Neuron
    {
        /// <summary>
        /// The neuron's value.
        /// </summary>
        public double Activation = 0;

        /// <summary>
        /// All the input connections and the source neurons
        /// </summary>
        public Dictionary<Neuron, Connection> Inputs = new Dictionary<Neuron, Connection>();

        /// <summary>
        /// Activation function delegate
        /// </summary>
        /// <param name="sumOfInputs"></param>
        /// <returns>Activation value</returns>
        public delegate double ActivationFunction(double sumOfInputs);

        /// <summary>
        /// The default activation function, which is a fast sigmoid-like function whose output varies between -1 and 1.
        /// </summary>
        public ActivationFunction ActivationFunc = DefActivationFunc;

        /// <summary>
        /// Used only by the parent Net object.
        /// </summary>
        public int Layer;
        /// <summary>
        /// Used only by the parent Net object.
        /// </summary>
        public int LayerNum;

        /// <summary>
        /// Used only by the parent Net object.
        /// </summary>
        public double x = 0;
        /// <summary>
        /// Used only by the parent Net object.
        /// </summary>
        public double y = 0;

        /// <summary>
        /// Used only by the parent Net object.
        /// </summary>
        public double InputsSum = 0;

        /// <summary>
        /// Used only by the parent Net object.
        /// </summary>
        public Net Net = null;

        /// <summary>
        /// Constructs the neuron
        /// </summary>
        /// <param name="net"></param>
        public Neuron()
        {
            Activation = 1;
        }

        /// <summary>
        /// The default activation function
        /// </summary>
        /// <param name="sumOfInputs"></param>
        /// <returns>Activation value</returns>
        static public double DefActivationFunc(double sumOfInputs)
        {
            return .5 * sumOfInputs / (1 + Math.Abs(sumOfInputs)) + .5;
        }

        /// <summary>
        /// Updates the Activation according to all the inputs.
        /// </summary>
        public void Iterate()
        {
            // don't calculate Activation for nodes that don't have inputs (like the input layer, and bias neurons)
            if(Inputs.Count < 1)
            {
                return;
            }

            InputsSum = 0;
            
            foreach(var c in Inputs)
            {
                InputsSum += c.Key.Activation * c.Value.Weight;
            }

            Activation = ActivationFunc(InputsSum);
        }
    }
}










