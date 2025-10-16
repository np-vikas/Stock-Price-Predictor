📈 Stock Price Predictor (React + TensorFlow.js)

A modern web application for predicting stock prices using an LSTM neural network built with TensorFlow.js — featuring auto-persistence, mock mode, and a sleek Tailwind-powered UI.

🚀 Features

LSTM Model Training — Build and train neural networks directly in your browser.

Auto-Fetch & Auto-Predict — Automatically loads previous ticker and predictions.

Persistent Model Storage — Models saved in IndexedDB for reuse.

Mock Mode — Simulated predictions for offline or TF-disabled environments.

Model Export / Import / Reset — Manage trained models easily.

Dark / Light Themes — Smooth theme switching via a slide-out settings drawer.

Responsive Visualization — Real-time charts (actual vs. predicted prices).

End-to-End Tested — Includes Jest + Playwright tests for reliability.

🧩 Prerequisites

Node.js (v18+ recommended)

npm or yarn

Alpha Vantage API key (for live stock data)

⚙️ Local Setup
# Clone the repository
git clone https://github.com/your-username/Stock-Price-Predictor.git
cd Stock-Price-Predictor

# Install dependencies
npm install

🧠 Running the App
npm run dev


Then open your browser and visit:
👉 http://localhost:5173
 (or the port shown in your terminal)

🔬 Enabling TensorFlow.js (Real Mode)

By default, the app starts in Mock Mode to ensure compatibility everywhere.

To enable TensorFlow.js:

Open the ⚙️ Settings Panel (top-right corner).

Toggle “Enable TensorFlow (Try)”.

Wait for TF.js to load (a small delay on first use).

Once loaded, train or load models as usual.

If your environment doesn’t support IndexedDB or GPU acceleration, the app will remain in mock mode automatically.

🧪 Running Tests
Unit Tests (Jest)
npm run test

End-to-End Tests (Playwright)
npx playwright install
npm run test:e2e

🧰 Build for Production
npm run build


Then serve it locally:

npm run preview

☁️ Deployment

You can easily deploy this app on Vercel or Netlify — both support client-side TensorFlow.js.

Example for Vercel:

vercel deploy

🧹 Troubleshooting
Issue	Fix
Build fails with TensorFlow.js errors	Use mock mode or delay TF.js load (already default)
IndexedDB errors	Clear browser storage and restart
Model not saving	Ensure you’re not in mock mode
Slow performance	Use fewer epochs or disable GPU backend
🧑‍💻 Author

Vikas Nomula
Built with ❤️ using React, Tailwind, and TensorFlow.js.
