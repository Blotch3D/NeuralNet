using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using NeuralNet;
using System.Runtime.InteropServices;

namespace TestNeuralNet
{
    /// <summary>
    /// Demonstrates how to use the Net class
    /// </summary>
    public partial class MainWindow : Window
    {
        Net Net;

        /// <summary>
        /// Holds training data
        /// </summary>
        List<(List<double> inputs, List<double> truths)> TrainingData;

        /// <summary>
        /// Used to periodically update graphical display
        /// </summary>
        System.Windows.Threading.DispatcherTimer DispatcherTimer = null;

        public MainWindow()
        {
            InitializeComponent();

            double LearnRate = 20;

            // Several solution types follow. Uncomment the one you want...

            /*
            // 3 bit ADC

            // Defines the number of neurons in each layer from left to right
            var layerInfo = new List<int>() { 1, 27, 3 };

            // several important arguments to the Net constructor and to Net.FindMinima
            var minWeight = -20;
            var maxWeight = 20;
            var costErrorBar = .3;

            TrainingData = new List<(List<double> inputs, List<double> truths)>()
            {
                (new List<double>() { 0 }, new List<double>() { .2, .2, .2 }),
                (new List<double>() { .1 }, new List<double>() { .2, .2, .8 }),
                (new List<double>() { .2 }, new List<double>() { .2, .8, .2 }),
                (new List<double>() { .3 }, new List<double>() { .2, .8, .8 }),
                (new List<double>() { .4 }, new List<double>() { .8, .2, .2 }),
                (new List<double>() { .5 }, new List<double>() { .8, .2, .8 }),
                (new List<double>() { .6 }, new List<double>() { .8, .8, .2 }),
                (new List<double>() { .7 }, new List<double>() { .8, .8, .8 }),
            };
            */
            // XOR

            // Defines the number of neurons in each layer from left to right
            var layerInfo = new List<int>() { 2, 4, 1 };

            // several important arguments to the Net constructor and to Net.FindMinima
            var minWeight = -8;
            var maxWeight = 8;
            var costErrorBar = 0;

            TrainingData = new List<(List<double> inputs, List<double> truths)>()
            {
                (new List<double>() { 0, 0 }, new List<double>() { .1 }),
                (new List<double>() { 1, 0 }, new List<double>() { .9 }),
                (new List<double>() { 0, 1 }, new List<double>() { .9 }),
                (new List<double>() { 1, 1 }, new List<double>() { .1 }),
            };

            /*
            // 4 input binary parity generator

            // Defines the number of neurons in each layer from left to right
            var layerInfo = new List<int>() { 4, 4, 1 };

            // several important arguments to the Net constructor and to Net.FindMinima
            var minWeight = -20;
            var maxWeight = 20;
            var costErrorBar = 0;

            TrainingData = new List<(List<double> inputs, List<double> truths)>()
            {
                (new List<double>() { 0, 0, 0, 0 }, new List<double>() { .1 }),
                (new List<double>() { 0, 0, 1, 0 }, new List<double>() { .9 }),
                (new List<double>() { 0, 0, 0, 1 }, new List<double>() { .9 }),
                (new List<double>() { 0, 0, 1, 1 }, new List<double>() { .1 }),
                (new List<double>() { 0, 1, 0, 0 }, new List<double>() { .9 }),
                (new List<double>() { 0, 1, 1, 0 }, new List<double>() { .1 }),
                (new List<double>() { 0, 1, 0, 1 }, new List<double>() { .1 }),
                (new List<double>() { 0, 1, 1, 1 }, new List<double>() { .9 }),
                (new List<double>() { 1, 0, 0, 0 }, new List<double>() { .9 }),
                (new List<double>() { 1, 0, 1, 0 }, new List<double>() { .1 }),
                (new List<double>() { 1, 0, 0, 1 }, new List<double>() { .1 }),
                (new List<double>() { 1, 0, 1, 1 }, new List<double>() { .9 }),
                (new List<double>() { 1, 1, 0, 0 }, new List<double>() { .1 }),
                (new List<double>() { 1, 1, 1, 0 }, new List<double>() { .9 }),
                (new List<double>() { 1, 1, 0, 1 }, new List<double>() { .9 }),
                (new List<double>() { 1, 1, 1, 1 }, new List<double>() { .1 }),
            };
            */

            /*
            // binary adder

            // Defines the number of neurons in each layer from left to right
            var layerInfo = new List<int>() { 4, 10, 10, 3 };

            // several important arguments to the Net constructor and to Net.FindMinima
            var minWeight = -20;
            var maxWeight = 20;
            var costErrorBar = 0;

            TrainingData = new List<(List<double> inputs, List<double> truths)>()
            {
                (new List<double>() { 0, 0, 0, 0 }, new List<double>() { .25, .25, .25 }),
                (new List<double>() { 0, 0, 0, 1 }, new List<double>() { .25, .25, .75 }),
                (new List<double>() { 0, 0, 1, 0 }, new List<double>() { .25, .75, .25 }),
                (new List<double>() { 0, 0, 1, 1 }, new List<double>() { .25, .75, .75 }),
                (new List<double>() { 0, 1, 0, 0 }, new List<double>() { .25, .25, .75 }),
                (new List<double>() { 0, 1, 0, 1 }, new List<double>() { .25, .75, .25 }),
                (new List<double>() { 0, 1, 1, 0 }, new List<double>() { .25, .75, .75 }),
                (new List<double>() { 0, 1, 1, 1 }, new List<double>() { .75, .25, .25 }),
                (new List<double>() { 1, 0, 0, 0 }, new List<double>() { .25, .75, .25 }),
                (new List<double>() { 1, 0, 0, 1 }, new List<double>() { .25, .75, .75 }),
                (new List<double>() { 1, 0, 1, 0 }, new List<double>() { .75, .25, .25 }),
                (new List<double>() { 1, 0, 1, 1 }, new List<double>() { .75, .25, .75 }),
                (new List<double>() { 1, 1, 0, 0 }, new List<double>() { .25, .75, .75 }),
                (new List<double>() { 1, 1, 0, 1 }, new List<double>() { .75, .25, .25 }),
                (new List<double>() { 1, 1, 1, 0 }, new List<double>() { .75, .25, .75 }),
                (new List<double>() { 1, 1, 1, 1 }, new List<double>() { .75, .75, .25 }),
            };
            */

            // we run the net in a separate thread
            var thread = new Thread(() =>
            {
                Net = new Net(layerInfo, minWeight, maxWeight);
                var best = Net.FindMinima(TrainingData, LearnRate, .98, 500000, 500, costErrorBar, 1.000001, .5, ShowNetQuality, 12);

                Net.SetWeights(best.weights);
                ShowNetQuality(Net);
                Console.WriteLine($"\n!!!   Learning complete.  BestCost = {best.cost} !!!");
            });

            thread.Priority = ThreadPriority.Lowest;
            thread.Start();

            // periodically update the graphical display
            DispatcherTimer = new System.Windows.Threading.DispatcherTimer();
            DispatcherTimer.Tick += new EventHandler(dispatcherTimer_Tick);
            DispatcherTimer.Interval = TimeSpan.FromMilliseconds(500);
            DispatcherTimer.Start();
        }

        private void dispatcherTimer_Tick(object sender, EventArgs e)
        {
            Background = new ImageBrush(Net.CreateBitmapImage());
        }

        /// <summary>
        /// Shows current net stats while its learning
        /// </summary>
        /// <param name="net"></param>
        void ShowNetQuality(Net net)
        {
            Console.WriteLine($"BestCost = {net.GlobalBestAvgCost}  hash={net.GetHashCode()}                                            ");

            foreach (var tData in TrainingData)
            {
                net.SetInput(tData.inputs);
                net.ForwardPropagate();

                var outputsString = "";
                var stringpositions = "".PadRight(170).ToArray();
                var outps = net.GetOutput();
                char n = '0';
                foreach (var o in outps)
                {
                    outputsString += $"{o:#.###}, ";
                    stringpositions[(int)(o * 169)] = n;
                    n++;
                }

                var s = string.Join("", stringpositions);

                Console.Write($"i={string.Join(", ", tData.inputs)}  o={outputsString} {s.ToString()}\n");
            }
        }

        private void MenuItem_Click(object sender, RoutedEventArgs e)
        {
            Net.AbortFindMinima();
        }
    }
}


