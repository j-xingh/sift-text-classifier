import { useState } from "react";
import { ArrowUpRight, RotateCcw, ShieldCheck } from "lucide-react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/predict";
/* ============================================================
   SIFT LOGO
   Signal lines pass through a filter and resolve into an "S".
   ============================================================ */

function SiftMark() {
  return (
    <div className="sift-mark" aria-label="SIFT logo" role="img">
      <svg viewBox="0 0 48 48" aria-hidden="true">
        <rect x="1" y="1" width="46" height="46" rx="4" className="logo-box" />
        <path
          d="M10 14h18M10 20h11M10 26h22M10 32h14M10 38h20"
          className="logo-lines"
        />
        <path d="M31 12v6h-6M31 18l-7 7 7 7h-7" className="logo-signal" />
        <circle cx="35.5" cy="14" r="2.2" className="logo-dot" />
        <circle cx="35.5" cy="34" r="2.2" className="logo-dot" />
      </svg>
    </div>
  );
}

/* ============================================================
   EMPTY / LOADING SIFT VISUAL
   ============================================================ */

function SiftGraphic({ active = false }) {
  return (
    <div className={`sift-graphic ${active ? "active" : ""}`} aria-hidden="true">
      <span className="graphic-dot graphic-dot-one" />
      <span className="graphic-dot graphic-dot-two" />
      <span className="graphic-dot graphic-dot-three" />
      <span className="graphic-line graphic-line-one" />
      <span className="graphic-line graphic-line-two" />
      <span className="graphic-line graphic-line-three" />
    </div>
  );
}

/* ============================================================
   MAIN APP
   ============================================================ */

function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeMessage = async () => {
    if (!message.trim()) {
      setError("Give me a message to sift first.");
      return;
    }

    setError("");
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error("Inference failed.");
      }

      const data = await response.json();
      setResult(data);
    } catch {
      setError("I can't reach the classifier. Make sure FastAPI is running.");
    } finally {
      setLoading(false);
    }
  };

  const clearMessage = () => {
    setMessage("");
    setResult(null);
    setError("");
  };

  const isSpam = result?.prediction === "spam";
  const probability = result ? Number(result.spam_probability) * 100 : 0;

  return (
    <main className="app-shell">
      <header className="site-header">
        <div className="brand">
          <SiftGraphic />
          <div className="brand-copy">
            <div className="brand-name">SIFT</div>
            <div className="brand-subtitle">
              Statistical Intelligence for Text Filtering
            </div>
          </div>
        </div>

        <div className="system-state" aria-label="SIFT system online">
          <span className="state-dot" />
          <span>Screening</span>
          <strong>Online</strong>
        </div>
      </header>

      <section className="intro">
        <div className="intro-content">
          <div className="intro-label">MESSAGE INTELLIGENCE</div>

          <h1>
            Should you trust
            <br />
            <span>this message?</span>
          </h1>

          <p>Drop it in. SIFT will look for the patterns.</p>
        </div>
      </section>

      <section className="workspace">
        <div className="message-panel">
          <div className="panel-header">
            <div className="panel-name">
              <span>01</span>
              Message
            </div>
            <div className="panel-meta">{message.length} / 2000</div>
          </div>

          <div className="input-frame">
            <textarea
              value={message}
              maxLength={2000}
              spellCheck="false"
              onChange={(event) => {
                setMessage(event.target.value);
                setError("");
              }}
              placeholder="Paste something suspicious here..."
              aria-label="Message to classify"
            />

            <div className="input-footer">
              <span>Plain text</span>
              <span>{message.length} characters</span>
            </div>
          </div>

          <div className="action-row">
            <button
              className="analyze-button"
              onClick={analyzeMessage}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="loader" />
                  Sifting...
                </>
              ) : (
                <>
                  Sift message
                  <ArrowUpRight size={16} strokeWidth={1.8} />
                </>
              )}
            </button>

            {result && (
              <button className="clear-button" onClick={clearMessage}>
                <RotateCcw size={13} strokeWidth={1.8} />
                Clear
              </button>
            )}
          </div>

          {error && <div className="error">{error}</div>}
        </div>

        <div
          className={`result-panel ${
            result ? (isSpam ? "spam-state" : "ham-state") : ""
          }`}
        >
          <div className="panel-header">
            <div className="panel-name">
              <span>02</span>
              SIFT says
            </div>
            <div className="panel-meta">
              {loading ? "Analyzing" : result ? "Result" : "Waiting"}
            </div>
          </div>

          {!result && !loading && (
            <div className="empty-state">
              <SiftGraphic />
              <h2>Nothing to sift yet.</h2>
              <p>
                Paste a message on the left and SIFT will take a look at its
                patterns.
              </p>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <SiftGraphic active />
              <h2>Sifting through it...</h2>
              <p>Turning the message into a statistical pattern.</p>
            </div>
          )}

          {result && (
            <div className="result-content">
              <div className="result-kicker">Classification complete</div>

              <div className="result-label">
                {isSpam ? "SPAM" : "LIKELY HAM"}
              </div>

              <div className="result-probability">
                <strong>
                  {probability.toFixed(1)}
                  <small>%</small>
                </strong>
                <span>spam likelihood</span>
              </div>

              <div
                className="probability-track"
                aria-label={`${probability.toFixed(1)} percent spam likelihood`}
              >
                <div
                  className="probability-fill"
                  style={{ width: `${Math.min(Math.max(probability, 0), 100)}%` }}
                />
              </div>

              <div className="result-message">
                <ShieldCheck size={17} strokeWidth={1.7} />
                <p>
                  {isSpam
                    ? "Strong spam signal detected. SIFT found patterns commonly associated with unsolicited communication."
                    : "No strong spam signal detected. SIFT did not find enough evidence to classify this message as spam."}
                </p>
              </div>

              <div className="analyzed">
                <span>You gave SIFT</span>
                <blockquote>“{message}”</blockquote>
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="engine" aria-label="Model information">
        <div>
          <span className="engine-title">SIFT ENGINE</span>
        </div>

        <div className="engine-details">
          <span>TF-IDF</span>
          <i />
          <span>LOGISTIC REGRESSION</span>
          <i />
          <span>5,169 SMS</span>
        </div>

        <div className="engine-score">
          TEST F1 <strong>84.12%</strong>
        </div>
      </section>

      <footer>
        <span>SIFT / 01</span>
        <span>Statistical Text Classification</span>
        <span>Local Inference</span>
      </footer>
    </main>
  );
}

export default App;
