
import { useEffect, useState } from "react";
import axios from "axios";

const API = "/api/v1";

const initialFeatures = {
  age: 42,
  income_k: 55,
  tenure_months: 24,
  past_purchases: 2,
  avg_basket: 40,
  region: "urban",
  is_member: 1,
};

export default function App() {
  const [status, setStatus] = useState(null);
  const [features, setFeatures] = useState(initialFeatures);
  const [dose, setDose] = useState(10);
  const [effect, setEffect] = useState(null);
  const [budget, setBudget] = useState(1000);
  const [margin, setMargin] = useState(0.3);
  const [limit, setLimit] = useState(1000);
  const [optimization, setOptimization] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get(`${API}/retail/model/status`)
      .then((response) => setStatus(response.data))
      .catch(() =>
        setError(
          "Could not connect to the API. Start the FastAPI backend and refresh."
        )
      );
  }, []);

  function updateFeature(name, value) {
    setFeatures((current) => ({ ...current, [name]: value }));
  }

  async function estimateEffect(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setEffect(null);

    try {
      const response = await axios.post(`${API}/retail/effect`, {
        features,
        dose: Number(dose),
      });
      setEffect(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail
          ? JSON.stringify(err.response.data.detail)
          : "Effect estimation failed. Check the API and model."
      );
    } finally {
      setLoading(false);
    }
  }

  async function optimizeBudget(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setOptimization(null);

    try {
      const response = await axios.post(`${API}/retail/optimize`, {
        budget: Number(budget),
        margin_rate: Number(margin),
        customer_limit: Number(limit),
      });
      setOptimization(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail
          ? JSON.stringify(err.response.data.detail)
          : "Budget optimization failed. Check the API and model."
      );
    } finally {
      setLoading(false);
    }
  }

  const money = (value) =>
    Number(value ?? 0).toLocaleString("en-IN", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 2,
    });

  const percent = (value) =>
    `${(Number(value ?? 0) * 100).toFixed(2)}%`;

  const number = (value) =>
    Number(value ?? 0).toLocaleString("en-IN", {
      maximumFractionDigits: 2,
    });

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span>EC</span> EconoCausal
        </div>

        <p className="side-label">WORKSPACE</p>

        <a className="nav-item active" href="#overview">
          Overview
        </a>
        <a className="nav-item" href="#effect">
          Causal estimator
        </a>
        <a className="nav-item" href="#optimizer">
          Budget optimizer
        </a>

        <div className="sidebar-note">
          <span className="status-dot" />
          Local development
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">CAUSAL MACHINE LEARNING</p>
            <h1>Dynamic Pricing Dashboard</h1>
            <p className="muted">
              Estimate discount effects and explore budget allocations.
            </p>
          </div>

          <span
            className={`pill ${
              status?.loaded ? "pill-ok" : "pill-pending"
            }`}
          >
            {status?.loaded ? "Model ready" : "Checking model"}
          </span>
        </header>

        {error && <div className="error-banner">{error}</div>}

        <section id="overview">
          <div className="section-heading">
            <div>
              <h2>Project overview</h2>
              <p className="muted">
                Retail model and current configuration
              </p>
            </div>
          </div>

          <div className="stats-grid">
            <article className="stat-card">
              <span className="stat-icon purple">ML</span>
              <p>Model status</p>
              <h3>{status?.loaded ? "Available" : "Unavailable"}</h3>
              <small>
                {status?.loaded ? "Retail DML model" : "Check backend"}
              </small>
            </article>

            <article className="stat-card">
              <span className="stat-icon blue">DML</span>
              <p>Method</p>
              <h3>Double ML</h3>
              <small>Continuous treatment</small>
            </article>

            <article className="stat-card">
              <span className="stat-icon orange">$</span>
              <p>Discount levels</p>
              <h3>
                {status?.supported_discount_levels?.join(" / ") || "N/A"}
              </h3>
              <small>Supported dose amounts</small>
            </article>
          </div>

          {status?.loaded && (
            <div className="notice">
              <strong>Development dataset</strong>
              <p>
                This model uses synthetic retail data. Its estimates are
                illustrative and are not validated business recommendations.
              </p>
            </div>
          )}
        </section>

        <section id="effect" className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">01 / CUSTOMER ANALYSIS</p>
              <h2>Causal effect estimator</h2>
              <p className="muted">
                Estimate conversion lift compared with zero discount.
              </p>
            </div>
          </div>

          <form onSubmit={estimateEffect}>
            <div className="form-grid">
              <label>
                Age
                <input
                  type="number"
                  min="18"
                  max="100"
                  required
                  value={features.age}
                  onChange={(e) =>
                    updateFeature("age", Number(e.target.value))
                  }
                />
              </label>

              <label>
                Income (thousands)
                <input
                  type="number"
                  min="0"
                  required
                  value={features.income_k}
                  onChange={(e) =>
                    updateFeature("income_k", Number(e.target.value))
                  }
                />
              </label>

              <label>
                Tenure (months)
                <input
                  type="number"
                  min="0"
                  required
                  value={features.tenure_months}
                  onChange={(e) =>
                    updateFeature("tenure_months", Number(e.target.value))
                  }
                />
              </label>

              <label>
                Past purchases
                <input
                  type="number"
                  min="0"
                  required
                  value={features.past_purchases}
                  onChange={(e) =>
                    updateFeature("past_purchases", Number(e.target.value))
                  }
                />
              </label>

              <label>
                Average basket value
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  required
                  value={features.avg_basket}
                  onChange={(e) =>
                    updateFeature("avg_basket", Number(e.target.value))
                  }
                />
              </label>

              <label>
                Region
                <select
                  value={features.region}
                  onChange={(e) =>
                    updateFeature("region", e.target.value)
                  }
                >
                  <option value="urban">Urban</option>
                  <option value="suburban">Suburban</option>
                  <option value="rural">Rural</option>
                </select>
              </label>

              <label>
                Membership
                <select
                  value={features.is_member}
                  onChange={(e) =>
                    updateFeature("is_member", Number(e.target.value))
                  }
                >
                  <option value={1}>Member</option>
                  <option value={0}>Non-member</option>
                </select>
              </label>

              <label>
                Discount amount
                <select
                  value={dose}
                  onChange={(e) => setDose(Number(e.target.value))}
                >
                  {(status?.supported_discount_levels || [0, 5, 10, 20]).map(
                    (level) => (
                      <option key={level} value={level}>
                        {level}
                      </option>
                    )
                  )}
                </select>
              </label>
            </div>

            <button
              className="primary-button"
              disabled={loading || !status?.loaded}
            >
              {loading ? "Processing..." : "Estimate causal effect"}
            </button>
          </form>

          {effect && (
            <div className="result-card effect-result">
              <p className="eyebrow">CAUSAL EFFECT ESTIMATE</p>

              <p className="muted">
                Selected discount: {dose}
              </p>

              <h3 className="effect-value">
                {percent(effect.estimated_conversion_lift)}
              </h3>

              <p className="effect-subtitle">
                Estimated conversion lift compared with zero discount
              </p>

              <div className="effect-details">
                <div>
                  <span>Discount amount</span>
                  <strong>{dose}</strong>
                </div>

                <div>
                  <span>Estimated lift (percentage points)</span>
                  <strong>
                    {(
                      Number(effect.estimated_conversion_lift ?? 0) * 100
                    ).toFixed(2)}
                  </strong>
                </div>
              </div>

              {effect.warnings?.length > 0 && (
                <div className="notice">
                  <strong>Warnings</strong>
                  <p>{effect.warnings.join(" ")}</p>
                </div>
              )}

              <p className="result-explanation">
                {effect.interpretation}
              </p>

              <small className="synthetic-note">
                Synthetic-data estimate. This is not a guaranteed conversion
                increase or a validated real-world business result.
              </small>
            </div>
          )}
        </section>

        <section id="optimizer" className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">02 / BUDGET ALLOCATION</p>
              <h2>Discount budget optimizer</h2>
              <p className="muted">
                Generate estimated discount recommendations for a customer
                sample.
              </p>
            </div>
          </div>

          <form onSubmit={optimizeBudget}>
            <div className="form-grid">
              <label>
                Total budget
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  required
                  value={budget}
                  onChange={(e) => setBudget(e.target.value)}
                />
              </label>

              <label>
                Margin rate (0 to 1)
                <input
                  type="number"
                  min="0.01"
                  max="1"
                  step="0.05"
                  required
                  value={margin}
                  onChange={(e) => setMargin(e.target.value)}
                />
              </label>

              <label>
                Customer sample size
                <input
                  type="number"
                  min="1"
                  max="20000"
                  required
                  value={limit}
                  onChange={(e) => setLimit(e.target.value)}
                />
              </label>
            </div>

            <button
              className="primary-button"
              disabled={loading || !status?.loaded}
            >
              {loading ? "Optimizing..." : "Optimize discount allocation"}
            </button>
          </form>

          {optimization && (
            <div className="optimization-results">
              <div className="result-grid">
                <div className="result-card budget-metric">
                  <p>Customers offered discounts</p>
                  <h3>
                    {number(
                      optimization.summary.customers_offered_discount
                    )}
                  </h3>
                  <small>Customers selected by the optimizer</small>
                </div>

                <div className="result-card budget-metric">
                  <p>Expected discount spend</p>
                  <h3>
                    {money(optimization.summary.expected_discount_spend)}
                  </h3>
                  <small>Of {money(budget)} available budget</small>
                </div>

                <div className="result-card budget-metric">
                  <p>Estimated incremental contribution</p>
                  <h3>
                    {money(
                      optimization.summary.estimated_incremental_contribution
                    )}
                  </h3>
                  <small>Model-estimated contribution</small>
                </div>

                <div className="result-card budget-metric">
                  <p>Budget remaining</p>
                  <h3>
                    {money(optimization.summary.budget_remaining)}
                  </h3>
                  <small>
                    {Number(budget) > 0
                      ? `${(
                          (Number(
                            optimization.summary.budget_remaining ?? 0
                          ) /
                            Number(budget)) *
                          100
                        ).toFixed(1)}% of available budget`
                      : "No budget available"}
                  </small>
                </div>
              </div>

              <h3 className="table-title">Recommendation preview</h3>

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Customer</th>
                      <th>Discount</th>
                      <th>Est. conversion</th>
                      <th>Expected spend</th>
                      <th>Est. contribution</th>
                    </tr>
                  </thead>

                  <tbody>
                    {optimization.recommendations_preview.map(
                      (row, index) => (
                        <tr key={row.customer_id ?? index}>
                          <td>{row.customer_id ?? "N/A"}</td>
                          <td>{row.recommended_discount}</td>
                          <td>
                            {percent(
                              row.estimated_conversion_probability
                            )}
                          </td>
                          <td>{money(row.expected_discount_spend)}</td>
                          <td>
                            {money(row.estimated_incremental_contribution)}
                          </td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>

              <p className="muted small">{optimization.note}</p>

              <p className="synthetic-note">
                Recommendations are generated using synthetic data and a
                heuristic optimizer. They should not be used for live pricing
                decisions without further validation.
              </p>
            </div>
          )}
        </section>

        <footer>
          EconoCausal | Educational prototype | Synthetic data | Not for live
          pricing decisions
        </footer>
      </main>
    </div>
  );
}
