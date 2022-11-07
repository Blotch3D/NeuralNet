using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNet
{
    /// <summary>
    /// Describes a connection. Connection objects are held in the Inputs dictionary.
    /// </summary>
    public class Connection
    {
        /// <summary>
        /// How much the output activation of the source (input) neuron is added to this neuron's Activation. Specifically, a
        /// multiplier of the input activation to generate a value added to the output neuron's input.
        ///
        /// Learning changes this, of course, but it will never go outside of MinWeight and MaxWeight.
        /// </summary>
        public double Weight = 0;

        /// <summary>
        /// Training will never let the weight go below this
        /// </summary>
        public double MinWeight = -1;

        /// <summary>
        /// Training will never let the weight go above this
        /// </summary>
        public double MaxWeight = 1;

        /// <summary>
        /// Training code can use this to keep track of how much Weight has changed from one configuration to the next.
        /// App code does not use this.
        /// </summary>
        public double Delta = 0;
    }
}
