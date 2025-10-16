import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { FiSettings } from 'react-icons/fi';

// Revised single-file React app with robust TensorFlow.js handling.
// Changes from previous version:
// - Removed automatic TF.js import at startup (avoids build errors when TF or IndexedDB unavailable).
// - Added an explicit "Enable TensorFlow" button in Settings that tries to load TF.js on demand.
// - Added a stable mock mode (default) so the UI + chart + training controls work even without TF.
// - Improved error handling and clear user-facing messages.

export default function StockPricePredictor() {
  const [symbol, setSymbol] = useState(localStorage.getItem('lastSymbol') || 'AAPL');
  const [apiKey, setApiKey] = useState(localStorage.getItem('lastApiKey') || '');
  const [rememberSettings, setRememberSettings] = useState(localStorage.getItem('rememberSettings') === 'true');
  const [data, setData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lstmParams, setLstmParams] = useState({ lookback: 20, units: 50, epochs: 30, batchSize: 8, lr: 0.001 });
  const [training, setTraining] = useState(false);
  const [lossHistory, setLossHistory] = useState([]);

  // TF / persistence state
  const [tfAvailable, setTfAvailable] = useState(false); // true when tfjs successfully loaded
  const [model, setModel] = useState(null); // tf.LayersModel when loaded/trained
  const [persistedModel, setPersistedModel] = useState(false);
  const [mockMode, setMockMode] = useState(true); // start in mock mode by default to avoid TF issues

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'light');

  // Simple check for IndexedDB availability (may be false in some sandboxes)
  const indexedDBAvailable = () => {
    try {
      return typeof indexedDB !== 'undefined' && indexedDB !== null;
    } catch (e) {
      return false;
    }
  };

  // Persist theme
  useEffect(() => {
    document.documentElement.className = theme;
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Auto-fetch last used symbol if user opted in
  useEffect(() => {
    if (rememberSettings && symbol && apiKey) {
      fetchStockData().catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // If user enables TF later, try loading persisted model
  const enableTensorFlow = async () => {
    try {
      const tf = await import('@tensorflow/tfjs');
      if (tf && tf.ready) await tf.ready();
      setTfAvailable(true);
      setMockMode(false);
      // try load persisted model if IndexedDB available
      if (indexedDBAvailable()) {
        try {
          const stored = await tf.loadLayersModel('indexeddb://lstm-stock-model');
          setModel(stored);
          setPersistedModel(true);
          console.log('Loaded persisted model from IndexedDB.');
        } catch (e) {
          console.warn('No persisted model found (or failed to load).');
        }
      }
      alert('TensorFlow.js loaded. Mock mode disabled.');
    } catch (err) {
      console.error('Failed to load TensorFlow.js:', err);
      alert('Unable to load TensorFlow.js in this environment. Continuing in mock mode.');
      setTfAvailable(false);
      setMockMode(true);
    }
  };

  const disableMockMode = () => {
    setMockMode(false);
  };

  const handleRememberToggle = (checked) => {
    setRememberSettings(checked);
    localStorage.setItem('rememberSettings', checked);
  };

  const fetchStockData = async () => {
    setLoading(true);
    try {
      if (rememberSettings) {
        localStorage.setItem('lastSymbol', symbol);
        localStorage.setItem('lastApiKey', apiKey);
      }
      const res = await axios.get(
        `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${encodeURIComponent(symbol)}&apikey=${encodeURIComponent(apiKey)}`
      );
      const raw = res.data?.['Time Series (Daily)'];
      if (!raw) throw new Error('Invalid response from API');
      const parsed = Object.keys(raw)
        .map((date) => ({ date, close: parseFloat(raw[date]['4. close']) }))
        .reverse();
      setData(parsed);
    } catch (err) {
      console.error(err);
      alert('Failed to fetch stock data. Check your API key or symbol.');
    } finally {
      setLoading(false);
    }
  };

  // Normalization helpers
  const normalize = (arr) => {
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const range = max - min || 1;
    return { norm: arr.map((v) => (v - min) / range), min, max };
  };
  const denormalize = (arr, min, max) => {
    const range = max - min || 1;
    return arr.map((v) => v * range + min);
  };

  // MOCK training: simulate training progress and produce a trivial model object
  const mockTrain = async () => {
    setTraining(true);
    setLossHistory([]);
    const epochs = Number(lstmParams.epochs || 10);
    const lossArr = [];
    for (let e = 0; e < epochs; e++) {
      // simulate loss decreasing
      const loss = Math.exp(-e / (epochs / 5)) + Math.random() * 0.05;
      lossArr.push(loss);
      setLossHistory([...lossArr]);
      // wait a bit to simulate training
      // eslint-disable-next-line no-await-in-loop
      await new Promise((r) => setTimeout(r, 200));
    }
    // create a tiny "mock model" object to indicate we have a model
    const mock = { mock: true, createdAt: Date.now() };
    setModel(mock);
    setPersistedModel(false);
    setTraining(false);
    alert('Mock training complete (TF not required).');
  };

  // MOCK predict: generate smooth synthetic predictions based on last close
  const mockPredict = async (horizon = 5) => {
    if (!data.length) return;
    const last = data[data.length - 1].close;
    const preds = [];
    for (let i = 1; i <= horizon; i++) {
      const jitter = (Math.sin(i) + Math.random() * 0.5) * (last * 0.01);
      preds.push(last + jitter);
    }
    const lastDate = new Date(data[data.length - 1].date);
    const futureDates = Array.from({ length: horizon }, (_, i) => {
      const d = new Date(lastDate);
      d.setDate(d.getDate() + i + 1);
      return d.toISOString().slice(0, 10);
    });
    setPredictions(futureDates.map((date, i) => ({ date, close: preds[i] })));
    alert('Mock predictions generated.');
  };

  // Train (real if TF available else mock)
  const trainLSTM = async () => {
    if (!data.length) return alert('Fetch data first.');
    if (mockMode || !tfAvailable) {
      await mockTrain();
      return;
    }

    // Real TF training path
    setTraining(true);
    setLossHistory([]);
    try {
      const tf = await import('@tensorflow/tfjs');
      if (tf && tf.ready) await tf.ready();
      const closes = data.map((d) => d.close);
      const { norm, min, max } = normalize(closes);
      const lookback = Number(lstmParams.lookback);
      const xs = [];
      const ys = [];
      for (let i = 0; i < norm.length - lookback; i++) {
        xs.push(norm.slice(i, i + lookback));
        ys.push(norm[i + lookback]);
      }
      if (xs.length === 0) throw new Error('Not enough data for the chosen lookback.');

      const xsTensor = tf.tensor2d(xs).reshape([xs.length, lookback, 1]);
      const ysTensor = tf.tensor2d(ys, [ys.length, 1]);

      const lstmModel = tf.sequential();
      lstmModel.add(tf.layers.lstm({ units: Number(lstmParams.units), inputShape: [lookback, 1] }));
      lstmModel.add(tf.layers.dense({ units: 1 }));
      lstmModel.compile({ optimizer: tf.train.adam(Number(lstmParams.lr)), loss: 'meanSquaredError' });

      const lossArr = [];
      await lstmModel.fit(xsTensor, ysTensor, {
        epochs: Number(lstmParams.epochs),
        batchSize: Number(lstmParams.batchSize),
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            lossArr.push(logs.loss);
            setLossHistory([...lossArr]);
            await tf.nextFrame();
          },
        },
      });

      if (indexedDBAvailable()) {
        try {
          await lstmModel.save('indexeddb://lstm-stock-model');
          setPersistedModel(true);
        } catch (saveErr) {
          console.warn('Failed to save model to IndexedDB:', saveErr?.message || saveErr);
        }
      }

      setModel(lstmModel);
      alert('Training complete (real TF).');
    } catch (err) {
      console.error('Training error:', err);
      alert('Training failed: ' + (err?.message || err));
    } finally {
      setTraining(false);
    }
  };

  // Predict (real TF if available else mock)
  const predictLSTM = async (horizon = 5) => {
    if (!data.length) return;
    if (mockMode || !tfAvailable) {
      await mockPredict(horizon);
      return;
    }

    try {
      const tf = await import('@tensorflow/tfjs');
      if (tf && tf.ready) await tf.ready();

      // ensure model is present
      let useModel = model;
      if (!useModel) {
        if (indexedDBAvailable()) {
          try {
            useModel = await tf.loadLayersModel('indexeddb://lstm-stock-model');
            setModel(useModel);
            setPersistedModel(true);
          } catch (e) {
            console.warn('Failed to load model from IndexedDB for prediction.');
            return;
          }
        } else {
          return;
        }
      }

      const closes = data.map((d) => d.close);
      const { norm, min, max } = normalize(closes);
      let inputSeq = norm.slice(-Number(lstmParams.lookback));
      const preds = [];

      for (let i = 0; i < horizon; i++) {
        const input = tf.tensor2d([inputSeq]).reshape([1, Number(lstmParams.lookback), 1]);
        const out = useModel.predict(input);
        const raw = await (Array.isArray(out) ? out[0].data() : out.data());
        const val = raw[0];
        preds.push(val);
        inputSeq = [...inputSeq.slice(1), val];
        input.dispose && input.dispose();
        if (out.dispose) out.dispose();
        await tf.nextFrame();
      }

      const denormPreds = denormalize(preds, min, max);
      const lastDate = new Date(data[data.length - 1].date);
      const futureDates = Array.from({ length: horizon }, (_, i) => {
        const d = new Date(lastDate);
        d.setDate(d.getDate() + i + 1);
        return d.toISOString().slice(0, 10);
      });

      setPredictions(futureDates.map((date, i) => ({ date, close: denormPreds[i] })));
    } catch (err) {
      console.error('Prediction error:', err);
      alert('Prediction failed: ' + (err?.message || err));
    }
  };

  // Export/import/delete use dynamic tf if available; otherwise warn
  const exportModel = async () => {
    if (!model) return alert('Train a model first.');
    if (mockMode || !tfAvailable) return alert('Cannot export in mock mode or without TensorFlow.js.');
    try {
      const tf = await import('@tensorflow/tfjs');
      if (tf && tf.ready) await tf.ready();
      await model.save('downloads://lstm-stock-model');
      alert('Model exported as files (JSON + weights).');
    } catch (err) {
      console.error('Export error:', err);
      alert('Export failed: ' + (err?.message || err));
    }
  };

  const importModel = async (event) => {
    try {
      if (!event?.target?.files || event.target.files.length < 1) throw new Error('No files selected');
      if (mockMode) return alert('Cannot import model while in mock mode.');
      const tf = await import('@tensorflow/tfjs');
      if (tf && tf.ready) await tf.ready();
      const files = Array.from(event.target.files);
      const loadedModel = await tf.loadLayersModel(tf.io.browserFiles(files));
      setModel(loadedModel);
      if (indexedDBAvailable()) {
        try {
          await loadedModel.save('indexeddb://lstm-stock-model');
          setPersistedModel(true);
        } catch (saveErr) {
          console.warn('Failed to persist imported model to IndexedDB:', saveErr?.message || saveErr);
        }
      }
      alert('Model imported and saved to browser storage (if available).');
    } catch (err) {
      console.error('Import error:', err);
      alert('Import failed: ' + (err?.message || err));
    }
  };

  const deletePersistedModel = async () => {
    if (!indexedDBAvailable()) return alert('IndexedDB not available in this environment.');
    if (mockMode) return alert('No persisted real model when in mock mode.');
    try {
      const tf = await import('@tensorflow/tfjs');
      if (tf && tf.ready) await tf.ready();
      await tf.io.removeModel('indexeddb://lstm-stock-model');
      setPersistedModel(false);
      setModel(null);
      alert('Persisted model deleted.');
    } catch (err) {
      console.error('Delete error:', err);
      alert('Failed to delete persisted model: ' + (err?.message || err));
    }
  };

  const resetAllSettings = async () => {
    localStorage.clear();
    setSymbol('AAPL');
    setApiKey('');
    setRememberSettings(false);
    setTheme('light');
    setMockMode(true);
    setTfAvailable(false);
    try {
      await deletePersistedModel();
    } catch (e) {
      // ignore
    }
    alert('All settings reset to default.');
  };

  const chartData = [
    ...data.map((d) => ({ ...d, type: 'history' })),
    ...predictions.map((p) => ({ ...p, type: 'prediction' })),
  ];

  return (
    <div className={`p-6 max-w-6xl mx-auto transition-colors duration-300 ${theme === 'dark' ? 'bg-gray-900 text-white' : 'bg-white text-black'}`}>
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-3xl font-bold">Stock Price Predictor (LSTM)</h1>

        <div className="flex items-center gap-3">
          {mockMode && (
            <div className="text-sm text-red-600 font-semibold animate-pulse">Mock Mode Active</div>
          )}
          <button onClick={() => setSettingsOpen(!settingsOpen)} className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"><FiSettings size={24} /></button>
        </div>
      </div>

      <div className={`fixed top-0 right-0 h-full w-80 bg-white dark:bg-gray-800 shadow-lg p-6 z-50 transition-transform duration-300 transform ${settingsOpen ? 'translate-x-0' : 'translate-x-full'}`}>
        <h2 className="text-xl font-semibold mb-6">Settings</h2>

        <div className="mb-4">
          <label className="block mb-1 font-medium">Theme</label>
          <select value={theme} onChange={(e) => setTheme(e.target.value)} className="border rounded p-2 w-full transition-colors bg-gray-50 dark:bg-gray-700 text-black dark:text-white">
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </div>

        <div className="mb-4">
          <label className="block mb-1 font-medium">API Key</label>
          <input type="text" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className="border rounded p-2 w-full bg-gray-50 dark:bg-gray-700 text-black dark:text-white" placeholder="Enter API Key" />
        </div>

        <div className="mb-4 flex items-center">
          <input type="checkbox" id="remember" checked={rememberSettings} onChange={(e) => handleRememberToggle(e.target.checked)} className="mr-2" />
          <label htmlFor="remember">Remember settings</label>
        </div>

        <div className="mb-4 flex items-center">
          <input type="checkbox" id="mock" checked={mockMode} onChange={(e) => setMockMode(e.target.checked)} className="mr-2" />
          <label htmlFor="mock">Enable Mock Mode</label>
        </div>

        <div className="flex flex-col gap-2">
          <button onClick={enableTensorFlow} className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold p-2 rounded transition-colors">Enable TensorFlow (try)</button>
          <button onClick={resetAllSettings} className="bg-red-500 hover:bg-red-600 text-white font-semibold p-2 rounded transition-colors">Reset All</button>
          <button onClick={trainLSTM} disabled={training} className="bg-green-500 hover:bg-green-600 text-white font-semibold p-2 rounded transition-colors">{training ? 'Training...' : 'Train LSTM'}</button>
          <button onClick={() => predictLSTM(5)} className="bg-blue-500 hover:bg-blue-600 text-white font-semibold p-2 rounded transition-colors">Predict</button>
          <button onClick={exportModel} className="bg-purple-500 hover:bg-purple-600 text-white font-semibold p-2 rounded transition-colors">Export Model</button>
          <label className="bg-yellow-500 hover:bg-yellow-600 text-black font-semibold p-2 rounded text-center cursor-pointer transition-colors">Import Model<input type="file" accept=".json,.bin" onChange={importModel} className="hidden" /></label>
          <button onClick={deletePersistedModel} className="bg-gray-500 hover:bg-gray-600 text-white font-semibold p-2 rounded transition-colors">Delete Model</button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
        <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="Symbol" className="border rounded p-2 w-full transition-colors bg-gray-50 dark:bg-gray-700 text-black dark:text-white" />
        <input value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="API Key" className="border rounded p-2 w-full transition-colors bg-gray-50 dark:bg-gray-700 text-black dark:text-white" />
        <button onClick={fetchStockData} className="bg-blue-500 hover:bg-blue-600 text-white font-semibold p-2 rounded transition-colors">{loading ? 'Loading...' : 'Fetch Data'}</button>
      </div>

      {chartData.length > 0 && (
        <div className="w-full h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={["auto", "auto"]} />
              <Tooltip />
              <Legend />
              <Line dataKey="close" data={chartData.filter((d) => d.type === 'history')} name="Actual" stroke="#3b82f6" dot={false} />
              <Line dataKey="close" data={chartData.filter((d) => d.type === 'prediction')} name="Predicted" stroke="#f97316" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {training && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Training Progress</h3>
          <p>Epochs: {lossHistory.length}/{lstmParams.epochs}</p>
          <p>Latest Loss: {lossHistory[lossHistory.length - 1]?.toFixed(6)}</p>
        </div>
      )}
    </div>
  );
}
