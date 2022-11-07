/*
IDEAS:

FindMinima becomes StartFindMinima, and doesn't block.

Add a StopFindMinima method. Remove AbortFindMinima

Add a 'GetBestCfg' method

During backpropagation, when traversing a neuron, calculate the differential of the activation function by
slightly perturbing it's input, then multiply the error by the magnitude of that differential.

Create and compile a forwardpropagate and a backpropagate equation rather than doing the operations with code.

*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Drawing;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.IO;
using System.Drawing.Imaging;

namespace NeuralNet
{
    /// <summary>
    /// A neural network.
    ///
    /// An initial (fully-connected) network shape can be specified to the constructor. After construction, you can
    /// alter it to any type of network you want. Call GetNeuron to get a neuron so you can alter, add, or delete any of
    /// its input connections or its bias. Call AddNeuron to add a neuron to a layer. Call DeleteNeuron to delete a
    /// neuron from a layer.
    ///
    /// To simplify everything, neuron bias values are really just another weighted input from a constant 'BiasNeuron'
    /// that has a constant activation of '1'.
    ///
    /// Train the network with the 'FindMinima' method. FindMinima takes several optimization parameters and runs
    /// multiple threads, where each thread runs multiple epochs with a cooling-down learn rate, backpropagation,
    /// periodically randomizing weights and resetting learn rate in order to jump out of local minima, and all the
    /// while keeping track of the best (lowest cost) network configuration. FindMinima efficiently uses all the CPU (we
    /// do not use GPU). Call AbortFindMinima (from another thread) to abort the last call to FindMinima.
    ///
    /// Visualization methods provide a Bitmap or BitmapImage of the current network structure, weights, and
    /// activation levels. Get and set the global network state with GetWeights and SetWeights. Use the
    /// SetInputs, ForwardPropagate, and GetOutputs methods to run the network.
    /// </summary>
    public class Net: ICloneable
    {
        /// <summary>
        /// Set this to see informative console messages about training progress.
        /// </summary>
        public bool DebugTrainingProgress = true;

        /// <summary>
        /// Set this to see informative console messages about lower level tasks.
        /// </summary>
        public bool DebugDetails = false;

        /// <summary>
        /// All neurons are held in this member. Each element of the list is a list of neurons for a layer,
        /// where the first element is a list of the neurons in the first layer, etc.. You can use this member
        /// to access any neuron so that you can read its activation value, override its default activation
        /// function, and (by using the neuron's 'Inputs' member) access its inputs to set individual weight
        /// training limits or even follow the path backward to a neuron in the previous layer. You may not,
        /// however, delete or add neurons or layers. Note that the 'bias' neurons (see class description) are
        /// included in this.</summary>
        public List<List<Neuron>> Neurons = new List<List<Neuron>>();

        List<int> LayerSizes = null;
        double MinWeight = -10;
        double MaxWeight = 10;
        Neuron.ActivationFunction ActivationFunction = null;

        Neuron BiasNeuron = new Neuron() { Activation = 1 };

        bool Abort = false;

        public double GlobalBestAvgCost = 1e300;
        public object LockGlobalBestAvgCost = new object();

        /// <summary>
        /// Constructs the feed forward neural network.
        /// </summary>
        /// <param name="layerSizes">A list indicating the number of neurons in each layer (i.e. the layer
        /// 'width'). The first (zeroth) element is how many neurons in the first (input) layer, and so on.
        /// (Layer widths should NOT include the bias neuron--it will be added automatically.)</param>
        /// <param name="minWeight">Default training lower weight limit. After construction, you can change
        /// this per-neuron by using the Neurons member.</param>
        /// <param name="maxWeight">Default training upper weight limit. After construction, you can change
        /// this per-neuron by using the Neurons member.</param>
        /// <param name="activationFunction">Default activation function. A value of null indicates the
        /// default function, which is '1/(1-abs(input))'. After construction, you can change this per-neuron
        /// by using the Neurons member.</param>
        public Net(
            List<int> layerSizes,
            double minWeight = -10,
            double maxWeight = 10,
            Neuron.ActivationFunction activationFunction = null)
        {
            LayerSizes = layerSizes.ToList();
            MinWeight = minWeight;
            MaxWeight = maxWeight;
            ActivationFunction = activationFunction;

            for(int layerNum = 0; layerNum < layerSizes.Count; layerNum++)
            {
                var numNeurons = layerSizes[layerNum];

                AddLayer
                (
                    numNeurons,
                    minWeight,
                    maxWeight,
                    activationFunction
                );
            }

            RandomizeWeights();
        }

        /// <summary>
        /// Given a Bitmap, return a BitmapImage
        /// </summary>
        /// <param name="bmp"></param>
        /// <returns></returns>
        public BitmapImage BitmapToBitmapImage(System.Drawing.Bitmap bmp)
        {
            var bitmapImage = new BitmapImage();

            using (var memory = new MemoryStream())
            {
                bmp.Save(memory, ImageFormat.Png);
                memory.Position = 0;

                bitmapImage.BeginInit();
                bitmapImage.StreamSource = memory;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
                bitmapImage.Freeze();
            }

            return bitmapImage;
        }

        /// <summary>
        /// Returns a graphical visualization of the network as a BitmapImage.
        /// </summary>
        /// <param name="width">Pixel width of the image</param>
        /// <param name="height">Pixel height of the image</param>
        /// <returns>A graphical visualization of the network</returns>
        public BitmapImage CreateBitmapImage(int width = 2048, int height = 1024)
        {
            var bmp = CreateBitmap(width, height);
            var bmpImg = BitmapToBitmapImage(bmp);
            bmp.Dispose();
            return bmpImg;
        }

        /// <summary>
        /// Returns a graphical visualization of the network as a Bitmap. The caller must call the bitmap's
        /// Dispose method when it is done using the bitmap.
        /// </summary>
        /// <param name="width">Pixel width of the image</param>
        /// <param name="height">Pixel height of the image</param>
        /// <returns>A graphical visualization of the network</returns>
        public Bitmap CreateBitmap(int width = 2048, int height = 1024)
        {
            var bmp = new Bitmap(width, height);
            var g = System.Drawing.Graphics.FromImage(bmp);
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

            var whiteBrush = new SolidBrush(System.Drawing.Color.FromArgb(255, 255, 255));
            var bkgndBrush = new SolidBrush(System.Drawing.Color.FromArgb(0, 0, 32));
            var font = new Font("Arial", 8);

            g.FillRectangle(bkgndBrush, 0, 0, width, height);

            var maxNeuronsPerLayer = 0;
            foreach(var neurons in Neurons)
            {
                var layerSize = neurons.Count();
                if (maxNeuronsPerLayer < layerSize)
                {
                    maxNeuronsPerLayer = layerSize;
                }
            }

            var sepx = width / Neurons.Count;
            var sepy = height / maxNeuronsPerLayer;

            var neuSize = 16;

            // calculate the location of all neurons
            var layerNum = 0;
            foreach (var neurons in Neurons)
            {
                layerNum++;
                var x = sepx * layerNum - sepx / 2;
                var y = (int)(height / 2 - (double)(neurons.Count - 1) / 2 * sepy);

                foreach (var neu in neurons)
                {
                    neu.x = x;
                    neu.y = y;
                    y += sepy;
                }
            }

            var txtPos = 0;

            // draw the connection lines
            foreach (var layer in Neurons)
            {
                foreach (var neu in layer)
                {
                    foreach (var connection in neu.Inputs)
                    {
                        if (connection.Key != BiasNeuron)
                        {
                            System.Drawing.Pen pen;
                            var color = connection.Value.Weight * 255 / (connection.Value.MaxWeight - connection.Value.MinWeight);
                            if (color > 255) color = 255;
                            if (color < -255) color = -255;
                            if (color > 0)
                            {
                                pen = new System.Drawing.Pen(System.Drawing.Color.FromArgb(0, (int)color, 0), 1);
                            }
                            else
                            {
                                pen = new System.Drawing.Pen(System.Drawing.Color.FromArgb((int)-color, 0, 0), 1);
                            }
                            g.DrawLine(pen, (int)neu.x, (int)neu.y, (int)connection.Key.x, (int)connection.Key.y);
                            pen.Dispose();
                        }
                    }
                }
            }


            // draw the connection and bias text
            foreach (var layer in Neurons)
            {
                foreach (var neu in layer)
                {
                    foreach (var connection in neu.Inputs)
                    {

                        if (connection.Key == BiasNeuron)
                        {
                            g.DrawString($"[{connection.Value.Weight:0.###}]", font, whiteBrush, (int)neu.x - 10, (int)neu.y + neuSize - 5);
                        }
                        else
                        {
                            System.Drawing.SolidBrush brush;
                            if (connection.Value.Weight > 0)
                            {
                                brush = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(0, 255, 0));
                            }
                            else
                            {
                                brush = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(255, 0, 0));
                            }

                            double x = 0;
                            double y = 0;

                            switch (txtPos)
                            {
                                case 0:
                                    x = (7 * neu.x + 1 * connection.Key.x) / 8 - 15;
                                    y = (7 * neu.y + 1 * connection.Key.y) / 8 - 9;
                                    break;
                                case 1:
                                    x = (6 * neu.x + 2 * connection.Key.x) / 8 - 15;
                                    y = (6 * neu.y + 2 * connection.Key.y) / 8 - 9;
                                    break;
                                case 2:
                                    x = (5 * neu.x + 3 * connection.Key.x) / 8 - 15;
                                    y = (5 * neu.y + 3 * connection.Key.y) / 8 - 9;
                                    break;
                                case 3:
                                    x = (4 * neu.x + 4 * connection.Key.x) / 8 - 15;
                                    y = (4 * neu.y + 4 * connection.Key.y) / 8 - 9;
                                    break;
                                case 4:
                                    x = (3 * neu.x + 5 * connection.Key.x) / 8 - 15;
                                    y = (3 * neu.y + 5 * connection.Key.y) / 8 - 9;
                                    break;
                                case 5:
                                    x = (2 * neu.x + 6 * connection.Key.x) / 8 - 15;
                                    y = (2 * neu.y + 6 * connection.Key.y) / 8 - 9;
                                    break;
                                case 6:
                                    x = (1 * neu.x + 7 * connection.Key.x) / 8 - 15;
                                    y = (1 * neu.y + 7 * connection.Key.y) / 8 - 9;
                                    break;
                            }

                            txtPos++;
                            txtPos %= 7;

                            g.DrawString($"{connection.Value.Weight:0.###}", font, brush, (int)x, (int)y);

                            brush.Dispose();
                        }
                    }
                }
            }

            // draw the neurons and their activation values
            foreach (var layer in Neurons)
            {
                foreach (var neu in layer)
                {
                    var x = (int)(neu.x - neuSize / 2);
                    var y = (int)(neu.y - neuSize / 2);

                    System.Drawing.Brush brush;
                    var color = neu.Activation * 255;
                    if (color > 255) color = 255;
                    if (color < -255) color = -255;
                    if (color > 0)
                    {
                        brush = new SolidBrush(System.Drawing.Color.FromArgb(0, (int)color, 0));
                    }
                    else
                    {
                        brush = new SolidBrush(System.Drawing.Color.FromArgb((int)-color, 0, 0));
                    }
                    g.FillEllipse(brush, x, y, neuSize, neuSize);
                    brush.Dispose();

                    g.DrawString($"{neu.Activation:0.###}", font, whiteBrush, x, y - neuSize);
                }
            }

            whiteBrush.Dispose();
            bkgndBrush.Dispose();
            font.Dispose();
            g.Dispose();

            return bmp;
        }

        /// <summary>
        /// Gets the neuron at the specified address
        /// </summary>
        /// <param name="layerNum">The zero-based layer number of the neuron</param>
        /// <param name="neuronNum">The number of the neuron in the layer. You can even specify one more than the number
        /// of neurons in the layer in order to get the 'bias' neuron used by the next layer.</param>
        /// <returns>The neuron at the specified address</returns>
        public Neuron GetNeuron(int layerNum, int neuronNum)
        {
            return Neurons[layerNum][neuronNum];
        }

        /// <summary>
        /// Deletes the neuron and its connections
        /// </summary>
        /// <param name="layerNum">The layer that this neuron lives in</param>
        /// <param name="neuronNum">Which neuron in the layer. You can even specify one more than the number of neurons
        /// in the layer in order to delete the 'bias' neuron used by the next layer.</param>
        public void DeleteNeuron(int layerNum,int neuronNum)
        {
            var neu = GetNeuron(layerNum, neuronNum);

            if(layerNum < Neurons.Count - 1)
            {
                foreach (var ne in Neurons[layerNum + 1])
                {
                    try
                    {
                        ne.Inputs.Remove(neu);
                    }
                    catch { }
                }
            }
        }

        /// <summary>
        /// Adds a neuron to the end of the layer, and creates input connections from all neurons in the previous layer,
        /// if any, and also adds an input connection from the bias neuron.
        /// </summary>
        /// <param name="layerNum">Which layer to add the neuron to</param>
        /// <param name="minWeight">Minimum allowed input connection weight</param>
        /// <param name="maxWeight">Maximum allowed input connection weight</param>
        /// <param name="activationFunction">The activation function, or null means use the default</param>
        public void AddNeuron(
            int layerNum,
            double minWeight = -10,
            double maxWeight = 10,
            Neuron.ActivationFunction activationFunction = null)
        {
            if (activationFunction == null)
            {
                activationFunction = Neuron.DefActivationFunc;
            }

            var neu = new Neuron()
            {
                Net = this,
                Layer = layerNum,
                LayerNum = layerNum,
                ActivationFunc = activationFunction
            };

            var layer = Neurons[layerNum];

            // This keeps the bias neuron intact, if there is one
            var neuNum = layer.Count;
            if(layer.Count > 0)
            {
                neuNum--;
            }

            layer.Insert(neuNum, neu);

            neu.Layer = layerNum;
            neu.LayerNum = layerNum;

            neu.Activation = 0;

            Connection connection;

            // If not the first layer, add input connections from the neurons in the previous layer
            if (layerNum > 0)
            {
                // Add to this neuron connections from all the previous layer neurons
                for (var m = 0; m < Neurons[layerNum - 1].Count(); m++)
                {
                    var prevNeu = Neurons[layerNum - 1][m];

                    connection = new Connection()
                    {
                        MinWeight = minWeight,
                        MaxWeight = maxWeight,
                        Weight = 0
                    };

                    neu.Inputs.Add(prevNeu, connection);
                }
            }

            // also add an input connection from the bias neuron
            connection = new Connection()
            {
                MinWeight = minWeight,
                MaxWeight = maxWeight,
                Weight = 0
            };

            neu.Inputs.Add(BiasNeuron, connection);
        }

        /// <summary>
        /// Adds a layer to the end of the network.
        /// </summary>
        /// <param name="numNeurons">How many neurons are in the new layer.</param>
        /// <param name="rndSeed">The random object to use to generate initial activation values.</param>
        /// <param name="minWeight">For input connections, learning will never result in a connection weight below this. This is
        /// meaningless for the first layer.</param>
        /// <param name="maxWeight">For input connections, learning will never result in a connection weight above this. This is
        /// meaningless for the first layer.</param>
        /// <param name="activationFunction">The activation function. A value of null indicates '1/(1-abs(input))'. This is
        /// meaningless for the first layer.</param>
        void AddLayer(
            int numNeurons,
            double minWeight = -10,
            double maxWeight = 10,
            Neuron.ActivationFunction activationFunction = null)
        {
            Neurons.Add(new List<Neuron>());

            var layerNum = Neurons.Count() - 1;

            // Add the neurons
            for (var n = 0; n < numNeurons; n++)
            {
                AddNeuron(layerNum, minWeight, maxWeight, activationFunction);
            }
        }

        /// <summary>
        /// Sets the first layer's Activation values
        /// </summary>
        /// <param name="inputs">The input values.</param>
        public void SetInput(List<double> inputs)
        {
            if (DebugDetails) Console.WriteLine($"Inputs = {string.Join(", ", inputs)}");
            if(inputs.Count != Neurons[0].Count())
            {
                throw new Exception($"{nameof(SetInput)}: input data length does not equal input layer length less the bias neuron");
            }

            for (var n = 0; n < Neurons[0].Count(); n++)
            {
                var neu = Neurons[0][n];
                neu.Activation = inputs[n];
            }
        }

        /// <summary>
        /// Runs one forward iteration of the neural network. Specifically, this calls every neuron's Iterate
        /// method in each layer, starting with the specified layer and then moving forward. Note: Before
        /// calling this you should call SetInput. Afterward you can call GetOutput.
        /// </summary>
        /// <param name="startLayer">The first layer to process. Subsequent layers will also be processed.
        /// Default is '1'. (Don't bother starting at the zeroth layer because there is nothing to do until
        /// layer 1.)</param>
        public void ForwardPropagate(int startLayer = 1)
        {
            if (DebugDetails) Console.WriteLine($"ForwardPropagate");

            var procLayers = Neurons.Skip(startLayer).ToList();
            foreach(var layer in procLayers)
            {
                foreach (var neu in layer)
                {
                    neu.Iterate();
                }
            }
        }

        /// <summary>
        /// Search for multiple local minima by running epochs while cooling down the learn rate. When it
        /// becomes apparent that we probably have found a local minimum, we again randomize the weights and
        /// set the learn rate to the starting value in order to jump out of that local minimum and search for
        /// another. We also keep track of the best (lowest) cost network state at all times, even while
        /// cooling down. For each thread but the first this clones the network. After certain other criteria
        /// are met (see the parameters), this returns the best state. You can also abort this at any time
        /// with a call to AbortFindMinima.
        ///
        /// The following algorithm is used to train a net:
        ///
        /// BestState = null BestAvgCost = 1e100
        ///
        /// PrevAvgCost = 1e300
        ///
        /// for MaxMinimumSearchs: LearnRate = InitialLearnRate RandomizeWeights()
        ///
        ///     for MaxEpochsPerMinimumSearch: AvgCost = RunEpoch(LearnRate) LearnRate *= CooldownRate
        ///
        ///         if BestAvgCost > AvgCost BestState = GetState BestAvgCost = AvgCost
        ///
        ///         if Abs(PrevAvgCost - AvgCost) &lt MinCostChange break
        ///
        ///         if AvgCost &lt MaxAvgCost return
        ///
        ///         PrevAvgCost = AvgCost
        ///
        /// </summary>
        /// <param name="trainingData">A list of input-truth tuples to train the network. Each input and in
        /// each truth are a list of activation values.</param>
        /// <param name="initialLearnRate">When a new Epoch is begun, this is the initial learn rate.</param>
        /// <param name="cooldownRate">LearnRate is multiplied by this each epoch</param>
        /// <param name="maxMinima">How many local minima to search</param>
        /// <param name="maxEpochsPerMinimum">Never let a local minimum search exceed this many epochs</param>
        /// <param name="costErrorBar">When calculating the cost, if an output neuron's activation differs by
        /// less than this much from the desired (truth) value, then consider it a perfect match.</param>
        /// <param name="minCostChangeRate">If the ratio of cost from one epoch to the next is less than this,
        /// then abort that local minimum search</param>
        /// <param name="minLearnRate">If the learn rate is less than this, then abort the local minimum
        /// search</param>
        /// <param name="monitorBestState">A callback that is given the current state whenever it
        /// improves</param>
        /// <param name="numThreads">How many threads (and Net clones) to use</param>
        /// <returns>A tuple of the best cost and the best weights</returns>
        public (double cost, List<double> weights) FindMinima(
            List<(List<double> inputs, List<double> truths)> trainingData,
            double initialLearnRate = 10,
            double cooldownRate = .998,
            int maxMinima = int.MaxValue,
            int maxEpochsPerMinimum = 200,
            double costErrorBar = .1,
            double minCostChangeRate = 0,
            double minLearnRate = .1,
            Action<Net> monitorBestState = null,
            int numThreads = 8)
        {
            Console.WriteLine($"z_{GetHashCode()}  {Cost(trainingData, costErrorBar)}");
            Abort = false;

            GlobalBestAvgCost = 1e300;

            if(numThreads > maxMinima)
            {
                numThreads = maxMinima;
            }

            maxMinima = maxMinima / numThreads;

            Parallel.For(0, numThreads, (n) =>
            {
                //Console.WriteLine($"d{GetHashCode()}");
                var newnet = (Net)Clone();
                //Console.WriteLine($"e{newnet.GetHashCode()}");
                var bstState = newnet.FindMinimaSingleThreaded
                (
                    this,
                    trainingData,
                    initialLearnRate,
                    cooldownRate,
                    maxMinima,
                    maxEpochsPerMinimum,
                    costErrorBar,
                    minCostChangeRate,
                    minLearnRate,
                    monitorBestState
                );
            });

            (double cost, List<double> weights) best = (GlobalBestAvgCost, GetWeights());

            return best;
        }

        /// <summary>
        /// Call this to abort a call to FindMinima
        /// </summary>
        public void AbortFindMinima()
        {
            Abort = true;
        }

        /// <summary>
        /// Same as Train, but only one thread
        /// </summary>
        /// <param name="baseNet">The original Net object that created the cloned Net object that calls this.
        /// The original object contains the best state information and the abort flag</param>
        /// <param name="trainingData">Same as FindMinima</param>
        /// <param name="initialLearnRate">Same as FindMinima</param>
        /// <param name="cooldownRate">Same as FindMinima</param>
        /// <param name="maxMinima">Same as FindMinima</param>
        /// <param name="maxEpochsPerMinimum">Same as FindMinima</param>
        /// <param name="minCostChangeRate">Same as FindMinima</param>
        /// <param name="minLearnRate">Same as FindMinima</param>
        /// <param name="monitorBestState">Same as FindMinima</param>
        /// <returns></returns>
        (double cost, List<double>) FindMinimaSingleThreaded(
            Net baseNet,
            List<(List<double> inputs, List<double> truths)> trainingData,
            double initialLearnRate = 10,
            double cooldownRate = .998,
            int maxMinima = int.MaxValue,
            int maxEpochsPerMinimum = 200,
            double costErrorbar = .1,
            double minCostChangeRate = 0,
            double minLearnRate = .1,
            Action<Net> monitorBestState = null)
        {
            if (DebugDetails) Console.WriteLine($"Train...");

            var bestState = new List<double>();
            var bestAvgCost = 1e300;
            var prevAvgCost = 1e300;
            int epochNum = 0;

            Console.WriteLine($"d_{GetHashCode()}  {Cost(trainingData, costErrorbar)}");

            for (int minimaNum = 0; minimaNum < maxMinima; minimaNum++)
            {
                if (DebugDetails) Console.WriteLine($"RunMinimumSearch");

                var learnRate = initialLearnRate;

                for (epochNum = 0; epochNum < maxEpochsPerMinimum; epochNum++)
                {
                    var avgCost = RunEpoch(trainingData, learnRate, costErrorbar);
                    //Console.WriteLine($"e_{GetHashCode()}  {avgCost}");
                    learnRate *= cooldownRate;
                    if (bestAvgCost > avgCost)
                    {
                        bestState = GetWeights();
                        bestAvgCost = avgCost;

                        if (bestAvgCost < baseNet.GlobalBestAvgCost)
                        {
                            lock (baseNet.LockGlobalBestAvgCost)
                            {
                                if (bestAvgCost < baseNet.GlobalBestAvgCost)
                                {
                                    Console.WriteLine($"f{GetHashCode()}    {bestAvgCost}");

                                    baseNet.GlobalBestAvgCost = bestAvgCost;
                                    baseNet.SetWeights(bestState);

                                    Console.WriteLine($"g{baseNet.GetHashCode()}");

                                    if (monitorBestState != null)
                                    {
                                        monitorBestState(baseNet);
                                    }
                                }
                            }
                        }
                    }

                    if (epochNum != 0 && (Math.Max(prevAvgCost / avgCost, avgCost / prevAvgCost) < minCostChangeRate))
                    {
                        if (DebugDetails) Console.WriteLine($"  Aborted minimum search run because cost not changing enough");
                        break;
                    }

                    if(learnRate < minLearnRate)
                    {
                        if (DebugDetails) Console.WriteLine($"  Aborted minimum search run because learnRate < minLearnRate");
                        break;
                    }

                    if (baseNet.Abort)
                    {
                        if (DebugDetails) Console.WriteLine($"  Aborted minimum search run because of abort signal");
                        return (bestAvgCost, bestState);
                    }

                    prevAvgCost = avgCost;
                }
                if (DebugTrainingProgress)
                {
                    Console.Write($"Aborting 'local minimum' search #{minimaNum} after {epochNum + 1} epochs and learnrate {learnRate:#.####}                       \r");
                }

                RandomizeWeights();
            }

            return (bestAvgCost, bestState);
        }

        /// <summary>
        /// Runs one epoch (i.e. trains with all training data)
        /// </summary>
        /// <param name="trainingData">A list of inputs-truths tuples for use as training data</param>
        /// <param name="learnRate">Learn rate</param>
        /// <param name="calcAvgCost">If true, then also calculate the average cost, and return it</param>
        /// <returns>If calcAvgCost is true, then the average cost</returns>
        public double RunEpoch(List<(List<double> inputs, List<double> truths)> trainingData, double learnRate, double costErrorBar = .1)
        {
            if (DebugDetails) Console.WriteLine($"RunEpoch");

            if (costErrorBar >= 0)
            {
                var avgCost = 0.0;
                for (var n = 0; n < trainingData.Count; n++)
                {
                    var sample = trainingData[n];
                    SetInput(sample.inputs);
                    ForwardPropagate();
                    avgCost += Loss(trainingData[n].truths, costErrorBar);
                    BackPropagate(sample.truths, learnRate);
                }
                return avgCost;
            }
            else
            {
                for(int n = 0; n< trainingData.Count; n++)
                {
                    var sample = trainingData[n];
                    SetInput(sample.inputs);
                    ForwardPropagate();
                    BackPropagate(sample.truths, learnRate);
                }
                return -1;
            }
        }
        public double Cost(List<(List<double> inputs, List<double> truths)> trainingData, double costErrorBar = .1)
        {
            var avgCost = 0.0;
            for (var n = 0; n < trainingData.Count; n++)
            {
                var sample = trainingData[n];
                SetInput(sample.inputs);
                ForwardPropagate();
                avgCost += Loss(trainingData[n].truths, costErrorBar);
            }
            return avgCost;
        }

        /// <summary>
        /// Returns the loss of the given truth vector with the current output of the network. (Assumes
        /// current output is for that truth)
        /// </summary>
        /// <param name="truths">A list of truths</param>
        /// <param name="costErrorBar">Before squaring the output error of a neuron, subtract this much from
        /// it's magnitude (and limit the result to zero.</param>
        /// <returns></returns>
        public double Loss(List<double> truths, double costErrorBar = 0)
        {
            double avgLoss = 0.0;
            var outputs = GetOutput();
            for (int m = 0; m < outputs.Count; m++)
            {
                var elem = outputs[m];
                var dif = elem - truths[m];

                if (dif < 0)
                {
                    dif += costErrorBar;
                    if (dif > 0)
                    {
                        dif = 0;
                    }
                }
                else
                {
                    dif -= costErrorBar;
                    if (dif < 0)
                    {
                        dif = 0;
                    }
                }

                var avgL = dif * dif;

                avgLoss += avgL;
            }
            return avgLoss;
        }

        /// <summary>
        /// Follow paths backward and add weight correction values according to the output error. The current output state must be
        /// that from the input associated with the specified truth.
        /// </summary>
        /// <param name="truth">The truth value</param>
        /// <param name="learnRate">The learn rate</param>
        public void BackPropagate(List<double> truth, double learnRate)
        {
            if (DebugDetails) Console.WriteLine($"Backpropagate");

            // for every output neuron
            for (int n =0; n < Neurons.Last().Count; n++)
            {
                // get the neuron
                var neu = Neurons.Last()[n];

                // figure the error
                var delta = truth[n] - neu.Activation;

                // follow the connections (and correct the weights) backward
                BackPropagateFromNeuron(neu, delta * learnRate);
            }
        }

        /// <summary>
        /// Similar to BackPropagate, but only from the specified neuron backwards, and the error must be specified.
        /// </summary>
        /// <param name="neu">The neuron from which to start backpropagation</param>
        /// <param name="delta">How much the neuron's activation should be changed to make it correct</param>
        /// <param name="learnRate">The learn rate</param>
        public void BackPropagateFromNeuron(Neuron neu, double delta)
        {
            if(neu.Inputs.Count < 1)
            {
                return;
            }
            delta /= neu.Inputs.Count;

            // for every input
            foreach (var c in neu.Inputs)
            {
                c.Value.Weight += c.Key.Activation * delta;

                if(c.Value.Weight > c.Value.MaxWeight)
                {
                    c.Value.Weight = c.Value.MaxWeight;
                }
                else if (c.Value.Weight < c.Value.MinWeight)
                {
                    c.Value.Weight = c.Value.MinWeight;
                }

                BackPropagateFromNeuron(c.Key, delta);
            }
        }
        /// <summary>
        /// Gets the last layer's Activation values
        /// </summary>
        /// <returns>The last layer's activation values</returns>
        public List<double> GetOutput()
        {
            var lastLayer = Neurons.Count - 1;

            var outputs = new List<double>();

            for (var n = 0; n < Neurons[lastLayer].Count(); n++)
            {
                var neu = Neurons[lastLayer][n];
                outputs.Add(neu.Activation);
            }

            return outputs;
        }

        /// <summary>
        /// Randomizes all weights (within their max and min values).
        /// </summary>
        public void RandomizeWeights()
        {
            if (DebugDetails) Console.WriteLine($"RandomizeWeights");

            var rnd = new Random();
            foreach (var layer in Neurons)
            {
                foreach(var neu in layer)
                {
                    foreach(var cn in neu.Inputs)
                    {
                        var c = cn.Value;
                        c.Weight = rnd.NextDouble() * (c.MaxWeight - c.MinWeight) + c.MinWeight;
                    }
                }
            }
        }


        /// <summary>
        /// Calculates a hash code from the weights (ignores the current activation values)
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            double hash = 0;
            foreach (var layer in Neurons)
            {
                foreach (var neu in layer)
                {
                    foreach (var con in neu.Inputs)
                    {
                        hash += con.Value.Weight;
                        hash *= 1.235718011;
                    }
                }
            }
            hash = Math.Abs(hash);
            //var exp = Math.Log10(int.MaxValue / hash);
            while (hash < int.MaxValue)
            {
                hash *= 1000;
            }
            while (hash > int.MaxValue)
            {
                hash /= 10;
            }

            return (int)hash;
        }

        public bool IsEqual(Net net)
        {
            var equals = true;

            if (net.Neurons.Count != Neurons.Count)
            {
                equals = false;
                goto done;
            }

            for (int layerNum = 0; layerNum < Neurons.Count; layerNum++)
            {
                var layerNeurons = Neurons[layerNum];
                var newLayerNeurons = net.Neurons[layerNum];

                if(layerNeurons.Count != newLayerNeurons.Count)
                {
                    equals = false;
                    goto done;
                }

                for (int neuronNum = 0; neuronNum < layerNeurons.Count; neuronNum++)
                {
                    var neu = layerNeurons[neuronNum];
                    var newNeu = newLayerNeurons[neuronNum];

                    if(newNeu.Activation != neu.Activation)
                    {
                        equals = false;
                        goto done;
                    }

                    var newEnum = newNeu.Inputs.GetEnumerator();
                    foreach (var conn in neu.Inputs)
                    {
                        newEnum.MoveNext();
                        var newConn = newEnum.Current;

                        if (conn.Key.Activation != newConn.Key.Activation || newConn.Value.Weight != conn.Value.Weight)
                        {
                            equals = false;
                            goto done;
                        }
                    }
                }
            }

            done:

            return equals;
        }

        /// <summary>
        /// Get's all the weights as a flat list of doubles
        /// </summary>
        /// <returns>All the weights</returns>
        public List<double> GetWeights()
        {
            var weights = new List<double>();
            foreach (var layer in Neurons)
            {
                foreach (var neu in layer)
                {
                    foreach (var con in neu.Inputs)
                    {
                        weights.Add(con.Value.Weight);
                    }
                }
            }
            return weights;
        }

        /// <summary>
        /// Sets all the weights as a flat list of doubles
        /// </summary>
        /// <param name="weights">All the weights</param>
        public void SetWeights(List<double> weights)
        {
            var en = weights.GetEnumerator();
            foreach (var layer in Neurons)
            {
                foreach (var neu in layer)
                {
                    foreach (var con in neu.Inputs)
                    {
                        en.MoveNext();
                        con.Value.Weight = en.Current;
                    }
                }
            }
        }


        /// <summary>
        /// Clones the Net
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            var net = new Net(LayerSizes, MinWeight, MaxWeight, ActivationFunction);

            net.DebugDetails = DebugDetails;
            net.DebugTrainingProgress = DebugTrainingProgress;

            for(int layerNum = 0; layerNum < Neurons.Count; layerNum++)
            {
                var layerNeurons = Neurons[layerNum];
                var newLayerNeurons = net.Neurons[layerNum];

                for (int neuronNum = 0; neuronNum < layerNeurons.Count; neuronNum++)
                {
                    var neu = layerNeurons[neuronNum];
                    var newNeu = newLayerNeurons[neuronNum];

                    newNeu.Activation = neu.Activation;

                    var newEnum = newNeu.Inputs.GetEnumerator();
                    foreach(var conn in neu.Inputs)
                    {
                        newEnum.MoveNext();
                        var newConn = newEnum.Current;

                        newConn.Value.Weight = conn.Value.Weight;
                    }
                }
            }

            return net;
        }
    }
}
