
import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import Plot from "react-plotly.js";

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

const chartConfig = {
  responsive: true,
  displayModeBar: false,
  scrollZoom: false,
  doubleClick: false,
};

const chartFont = {
  family: "Inter, Segoe UI, sans-serif",
  size: 12,
  color: "#62677f",
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
  const [statusError, setStatusError] = useState("");

  // ------------------------------------------
  // LOAD MODEL STATUS
  // ------------------------------------------

  useEffect(() => {
    let active = true;

    axios
      .get(`${API}/retail/model/status`)
      .then((response) => {
        if (active) {
          setStatus(response.data);
          setStatusError("");
        }
      })
      .catch(() => {
        if (active) {
          setStatusError(
            "Could not connect to the backend. Start FastAPI and refresh."
          );
        }
      });

    return () => {
      active = false;
    };
  }, []);

  // ------------------------------------------
  // FORMATTERS
  // ------------------------------------------

  const money = (value) =>
    Number(value ?? 0).toLocaleString("en-IN", {
      style: "currency",
      currency: "INR",
      maximumFractionDigits: 2,
    });

  const number = (value) =>
    Number(value ?? 0).toLocaleString("en-IN", {
      maximumFractionDigits: 2,
    });

  const percent = (value) =>
    `${(Number(value ?? 0) * 100).toFixed(2)}%`;

  const percentagePoints = (value) =>
    `${(Number(value ?? 0) * 100).toFixed(2)} pp`;

  // ------------------------------------------
  // CUSTOMER FEATURE UPDATE
  // ------------------------------------------

  function updateFeature(name, value) {
    setFeatures((current) => ({
      ...current,
      [name]: value,
    }));

    // Remove stale results when customer details change.
    setEffect(null);
    setError("");
  }
  function validateCustomerFeatures() {
    const age = Number(features.age);
    const income = Number(features.income_k);
    const tenure = Number(features.tenure_months);
    const purchases = Number(features.past_purchases);
    const basket = Number(features.avg_basket);

    if (!Number.isFinite(age) || age < 18 || age > 80 || !Number.isInteger(age)) {
      return "Age must be a whole number between 18 and 80.";
    }

    if (!Number.isFinite(income) || income < 10 || income > 400) {
      return "Income must be between 10 and 400 thousand.";
    }

    if (
      !Number.isFinite(tenure) ||
      tenure < 0 ||
      tenure > 120 ||
      !Number.isInteger(tenure)
    ) {
      return "Tenure must be a whole number between 0 and 120 months.";
    }

    if (
      !Number.isFinite(purchases) ||
      purchases < 0 ||
      !Number.isInteger(purchases)
    ) {
      return "Past purchases must be a non-negative whole number.";
    }

    if (!Number.isFinite(basket) || basket < 8 || basket > 400) {
      return "Average basket value must be between 8 and 400.";
    }

    if (!["urban", "suburban", "rural"].includes(features.region)) {
      return "Please select a valid region.";
    }

    if (![0, 1].includes(Number(features.is_member))) {
      return "Please select a valid membership status.";
    }

    return "";
  }

  // ------------------------------------------
  // ESTIMATE CAUSAL EFFECT
  // ------------------------------------------

  async function estimateEffect(event) {
    event.preventDefault();

    // Validate customer details before sending the API request.
    const validationMessage = validateCustomerFeatures();

    if (validationMessage) {
      setEffect(null);
      setError(validationMessage);
      return;
    }

    setLoading(true);
    setError("");
    setEffect(null);

    try {
      const response = await axios.post(`${API}/retail/effect`, {
        features: {
          age: Number(features.age),
          income_k: Number(features.income_k),
          tenure_months: Number(features.tenure_months),
          past_purchases: Number(features.past_purchases),
          avg_basket: Number(features.avg_basket),
          region: features.region,
          is_member: Number(features.is_member),
        },
        dose: Number(dose),
      });

      setEffect(response.data);
    } catch (err) {
      const detail = err.response?.data?.detail;

      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg).join(" ")
        : typeof detail === "string"
          ? detail
          : "Effect estimation failed. Check the API and model.";

      setError(message);
    } finally {
      setLoading(false);
    }
  }

  // ------------------------------------------
  // OPTIMIZE DISCOUNT BUDGET
  // ------------------------------------------

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

  // ------------------------------------------
  // OPTIMIZATION SUMMARY
  // ------------------------------------------

  const summary = optimization?.summary;

  const expectedSpend = Math.max(
    0,
    Number(summary?.expected_discount_spend ?? 0)
  );

  const remainingBudget = Math.max(
    0,
    Number(summary?.budget_remaining ?? 0)
  );

  const availableBudget = Math.max(0, Number(budget) || 0);

  const budgetUtilization =
    availableBudget > 0
      ? (expectedSpend / availableBudget) * 100
      : 0;

  const recommendations = useMemo(
    () => optimization?.recommendations_preview ?? [],
    [optimization]
  );

  // ------------------------------------------
  // CHART 1: CAUSAL EFFECT
  // ------------------------------------------

  const effectLift =
    Number(effect?.estimated_conversion_lift ?? 0) * 100;

  const effectChartData = useMemo(() => {
    if (!effect) return [];

    return [
      {
        type: "bar",
        orientation: "h",
        x: [effectLift],
        y: ["Estimated conversion lift"],
        marker: {
          color: "#7562e9",
          line: {
            color: "#5945ce",
            width: 1,
          },
        },
        text: [percentagePoints(effectLift / 100)],
        textposition: "outside",
        cliponaxis: false,
        width: 0.42,
        hovertemplate:
          "<b>Estimated conversion lift</b><br>" +
          "%{x:.2f} percentage points<extra></extra>",
      },
    ];
  }, [effect, effectLift]);

  const effectChartLayout = {
    autosize: true,
    height: 290,
    margin: {
      t: 35,
      r: 75,
      b: 65,
      l: 145,
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: chartFont,
    xaxis: {
      title: {
        text: "Conversion lift (percentage points)",
        font: { size: 11 },
      },
      zeroline: true,
      zerolinecolor: "#aeb2c3",
      zerolinewidth: 2,
      gridcolor: "#eceef5",
      automargin: true,
      fixedrange: true,
    },
    yaxis: {
      automargin: true,
      fixedrange: true,
    },
    showlegend: false,
    hoverlabel: {
      bgcolor: "#24243e",
      bordercolor: "#24243e",
      font: {
        color: "#ffffff",
        size: 12,
      },
    },
  };

  // ------------------------------------------
  // CHART 2: BUDGET UTILIZATION
  // ------------------------------------------

  const budgetChartData = useMemo(() => {
    if (!optimization) return [];

    return [
      {
        type: "pie",
        labels: ["Expected discount spend", "Remaining budget"],
        values: [expectedSpend, remainingBudget],
        hole: 0.68,
        sort: false,
        direction: "clockwise",
        marker: {
          colors: ["#7562e9", "#e7e9f2"],
          line: {
            color: "#ffffff",
            width: 4,
          },
        },
        textinfo: "none",
        hovertemplate:
          "<b>%{label}</b><br>" +
          "Amount: ₹%{value:,.2f}<br>" +
          "Share: %{percent}<extra></extra>",
        showlegend: true,
      },
    ];
  }, [optimization, expectedSpend, remainingBudget]);

  const budgetChartLayout = {
    autosize: true,
    height: 310,
    margin: {
      t: 15,
      r: 15,
      b: 55,
      l: 15,
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: chartFont,
    showlegend: true,
    legend: {
      orientation: "h",
      x: 0.5,
      xanchor: "center",
      y: -0.08,
      font: {
        size: 10,
      },
    },
    annotations: [
      {
        text:
          `<b>${budgetUtilization.toFixed(1)}%</b>` +
          "<br><span style='font-size:10px'>Utilized</span>",
        x: 0.5,
        y: 0.5,
        showarrow: false,
        font: {
          size: 19,
          color: "#292544",
        },
      },
    ],
    hoverlabel: {
      bgcolor: "#24243e",
      bordercolor: "#24243e",
      font: {
        color: "#ffffff",
        size: 12,
      },
    },
  };

  // ------------------------------------------
  // CHART 3: DISCOUNT ALLOCATION
  // ------------------------------------------

  const discountLevels =
    status?.supported_discount_levels ?? [0, 5, 10, 20];

  const discountDistribution = discountLevels.map((level) =>
    recommendations.filter(
      (row) =>
        Number(row.recommended_discount ?? 0) === Number(level)
    ).length
  );

  const discountChartData = useMemo(() => {
    if (!recommendations.length) return [];

    return [
      {
        type: "bar",
        x: discountLevels.map((level) => `₹${level}`),
        y: discountDistribution,
        marker: {
          color: ["#c5befb", "#a89af7", "#8270ed", "#5945ce"],
          line: {
            color: "#ffffff",
            width: 1,
          },
        },
        text: discountDistribution.map((value) =>
          value > 0 ? String(value) : ""
        ),
        textposition: "outside",
        cliponaxis: false,
        hovertemplate:
          "<b>Discount: %{x}</b><br>" +
          "Customers: %{y}<extra></extra>",
      },
    ];
  }, [recommendations, discountLevels.join(","), discountDistribution.join(",")]);

  const discountChartLayout = {
    autosize: true,
    height: 310,
    margin: {
      t: 30,
      r: 20,
      b: 55,
      l: 60,
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: chartFont,
    xaxis: {
      title: {
        text: "Recommended discount",
        font: { size: 11 },
      },
      showgrid: false,
      fixedrange: true,
    },
    yaxis: {
      title: {
        text: "Customers",
        font: { size: 11 },
      },
      rangemode: "tozero",
      gridcolor: "#eceef5",
      fixedrange: true,
    },
    showlegend: false,
    hoverlabel: {
      bgcolor: "#24243e",
      bordercolor: "#24243e",
      font: {
        color: "#ffffff",
        size: 12,
      },
    },
  };

  // ------------------------------------------
  // DASHBOARD
  // ------------------------------------------

  return (
    <div className="app-shell">

      {/* SIDEBAR */}

      <aside className="sidebar">
        <div className="brand">
          <span className="brand-logo">EC</span>
          <span>EconoCausal</span>
        </div>

        <p className="side-label">WORKSPACE</p>

        <nav className="sidebar-nav">
          <a className="nav-item active" href="#overview">
            <span className="nav-symbol">01</span>
            Overview
          </a>

          <a className="nav-item" href="#effect">
            <span className="nav-symbol">02</span>
            Causal estimator
          </a>

          <a className="nav-item" href="#optimizer">
            <span className="nav-symbol">03</span>
            Budget optimizer
          </a>
        </nav>

        <div className="sidebar-note">
          <span className="status-dot" />
          Local development
        </div>
      </aside>

      {/* MAIN CONTENT */}

      <main className="main-content">

        {/* HEADER */}

        <header className="topbar">
          <div>
            <p className="eyebrow">CAUSAL MACHINE LEARNING PLATFORM</p>

            <h1>Dynamic Pricing Dashboard</h1>

            <p className="muted">
              Estimate customer-level discount effects and optimize
              marketing budget allocation.
            </p>
          </div>

          <div
            className={`pill ${status?.loaded ? "pill-ok" : "pill-pending"
              }`}
          >
            <span className="status-dot" />
            {status?.loaded ? "Model ready" : "Checking model"}
          </div>
        </header>

        {(error || statusError) && (
          <div className="error-banner">
            {error || statusError}
          </div>
        )}

        {/* OVERVIEW */}

        <section id="overview" className="overview-section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">PROJECT SUMMARY</p>
              <h2>Project overview</h2>
              <p className="muted">
                Model availability and configuration
              </p>
            </div>
          </div>

          <div className="stats-grid">

            <article className="stat-card">
              <span className="stat-icon purple">ML</span>
              <p>Model status</p>
              <h3>
                {status?.loaded ? "Available" : "Unavailable"}
              </h3>
              <small>
                {status?.loaded
                  ? "Retail DML model loaded"
                  : "Check backend connection"}
              </small>
            </article>

            <article className="stat-card">
              <span className="stat-icon blue">DML</span>
              <p>Causal method</p>
              <h3>Double ML</h3>
              <small>Continuous treatment estimation</small>
            </article>

            <article className="stat-card">
              <span className="stat-icon orange">₹</span>
              <p>Discount levels</p>
              <h3 className="discount-levels">
                {status?.supported_discount_levels?.join(" / ") || "N/A"}
              </h3>
              <small>Supported discount amounts</small>
            </article>

          </div>

          {status?.loaded && (
            <div className="notice">
              <strong>Development dataset</strong>
              <p>
                This project uses synthetic retail data. The results
                demonstrate causal estimation and budget optimization
                methods and are not validated live-business recommendations.
              </p>
            </div>
          )}
        </section>

        {/* CAUSAL EFFECT ESTIMATOR */}

        <section id="effect" className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">01 / CUSTOMER ANALYSIS</p>
              <h2>Causal effect estimator</h2>
              <p className="muted">
                Estimate the conversion lift for a selected customer
                compared with offering no discount.
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
                  max="80"
                  step="1"
                  required
                  value={features.age}
                  onChange={(e) =>
                    updateFeature(
                      "age",
                      e.target.value === "" ? "" : Number(e.target.value)
                    )
                  }
                />
              </label>

              <label>
                Income (thousands)
                <input
                  type="number"
                  min="10"
                  max="400"
                  step="0.1"
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
                  max="120"
                  step="1"
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
                  step="1"
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
                  min="8"
                  max="400"
                  step="0.01"
                  required
                  value={features.avg_basket}
                  onChange={(e) =>
                    updateFeature(
                      "avg_basket",
                      e.target.value === "" ? "" : Number(e.target.value)
                    )
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
                  onChange={(e) => {
                    setDose(Number(e.target.value));
                    setEffect(null);
                    setError("");
                  }}
                >
                  {(status?.supported_discount_levels ?? [0, 5, 10, 20]).map(
                    (level) => (
                      <option key={level} value={level}>
                        {money(level)}
                      </option>
                    )
                  )}
                </select>
              </label>

            </div>

            <button
              className="primary-button"
              type="submit"
              disabled={loading || !status?.loaded}
            >
              {loading ? "Processing..." : "Estimate causal effect"}
            </button>
          </form>

          {/* EFFECT RESULT */}

          {effect && (
            <div className="effect-result">

              <div className="effect-result-header">
                <div>
                  <p className="eyebrow">CAUSAL EFFECT ESTIMATE</p>
                  <p className="muted">
                    Selected discount: {money(dose)}
                  </p>
                </div>
                <span className="result-tag">DML estimate</span>
              </div>

              <h3 className="effect-value">
                {percent(effect.estimated_conversion_lift)}
              </h3>

              <p className="effect-subtitle">
                Estimated conversion lift compared with zero discount
              </p>

              <div className="effect-details">
                <div>
                  <span>Discount amount</span>
                  <strong>{money(dose)}</strong>
                </div>

                <div>
                  <span>Conversion lift</span>
                  <strong>
                    {percentagePoints(effect.estimated_conversion_lift)}
                  </strong>
                </div>
              </div>

              {effect.warnings?.length > 0 && (
                <div className="notice">
                  <strong>Model warnings</strong>
                  <p>{effect.warnings.join(" ")}</p>
                </div>
              )}

              {effect.interpretation && (
                <p className="result-explanation">
                  {effect.interpretation}
                </p>
              )}

              <div className="chart-panel effect-chart-panel">
                <div className="chart-heading">
                  <div>
                    <h3>Estimated causal effect</h3>
                    <p className="muted">
                      Predicted conversion lift for the selected customer
                    </p>
                  </div>
                </div>

                <div className="plot-container">
                  <Plot
                    data={effectChartData}
                    layout={effectChartLayout}
                    config={chartConfig}
                    style={{ width: "100%", height: "290px" }}
                    useResizeHandler
                  />
                </div>
              </div>

              <p className="synthetic-note">
                Synthetic-data estimate. This is not a guaranteed
                conversion increase or a validated real-world result.
              </p>

            </div>
          )}
        </section>

        {/* BUDGET OPTIMIZER */}

        <section id="optimizer" className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">02 / BUDGET ALLOCATION</p>
              <h2>Discount budget optimizer</h2>
              <p className="muted">
                Allocate discount offers under a specified expected-spend
                budget.
              </p>
            </div>
          </div>

          <form onSubmit={optimizeBudget}>
            <div className="form-grid optimizer-form-grid">

              <label>
                Total budget (₹)
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
                Contribution margin rate
                <input
                  type="number"
                  min="0.01"
                  max="1"
                  step="0.01"
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
              type="submit"
              disabled={loading || !status?.loaded}
            >
              {loading ? "Optimizing..." : "Optimize discount allocation"}
            </button>
          </form>

          {/* OPTIMIZATION RESULTS */}

          {optimization && (
            <div className="optimization-results">

              {/* SUMMARY METRICS */}

              <div className="result-grid">

                <article className="result-card budget-metric">
                  <p>Customers offered discounts</p>
                  <h3>
                    {number(summary?.customers_offered_discount)}
                  </h3>
                  <small>Selected by the optimizer</small>
                </article>

                <article className="result-card budget-metric">
                  <p>Expected discount spend</p>
                  <h3>{money(expectedSpend)}</h3>
                  <small>
                    Of {money(availableBudget)} available budget
                  </small>
                </article>

                <article className="result-card budget-metric">
                  <p>Estimated incremental contribution</p>
                  <h3>
                    {money(summary?.estimated_incremental_contribution)}
                  </h3>
                  <small>Model-estimated contribution</small>
                </article>

                <article className="result-card budget-metric">
                  <p>Budget remaining</p>
                  <h3>{money(remainingBudget)}</h3>
                  <small>
                    {availableBudget > 0
                      ? `${(
                        (remainingBudget / availableBudget) *
                        100
                      ).toFixed(1)}% of available budget`
                      : "No budget available"}
                  </small>
                </article>

              </div>

              {/* CHARTS */}

              <div className="charts-grid">

                {/* BUDGET DOUGHNUT */}

                <div className="chart-panel">
                  <div className="chart-heading">
                    <div>
                      <h3>Budget utilization</h3>
                      <p className="muted">
                        Expected spend versus remaining budget
                      </p>
                    </div>
                  </div>

                  {expectedSpend + remainingBudget > 0 ? (
                    <div className="plot-container">
                      <Plot
                        data={budgetChartData}
                        layout={budgetChartLayout}
                        config={chartConfig}
                        style={{ width: "100%", height: "310px" }}
                        useResizeHandler
                      />
                    </div>
                  ) : (
                    <div className="empty-chart">
                      <span>No budget allocation</span>
                      <p>
                        There is no expected discount spend to visualize.
                      </p>
                    </div>
                  )}
                </div>

                {/* DISCOUNT DISTRIBUTION */}

                <div className="chart-panel">
                  <div className="chart-heading">
                    <div>
                      <h3>Discount allocation</h3>
                      <p className="muted">
                        Customers by recommended discount level
                      </p>
                    </div>
                  </div>

                  {recommendations.length > 0 ? (
                    <div className="plot-container">
                      <Plot
                        data={discountChartData}
                        layout={discountChartLayout}
                        config={chartConfig}
                        style={{ width: "100%", height: "310px" }}
                        useResizeHandler
                      />
                    </div>
                  ) : (
                    <div className="empty-chart">
                      <span>No recommendations available</span>
                      <p>
                        The API did not return recommendation preview rows.
                      </p>
                    </div>
                  )}
                </div>

              </div>

              <p className="chart-footnote">
                Discount allocation visualization uses the recommendation
                preview returned by the API, not necessarily the complete
                optimized customer population.
              </p>

              {/* RECOMMENDATION TABLE */}

              <div className="table-section">
                <div className="table-heading">
                  <div>
                    <p className="eyebrow">OPTIMIZATION OUTPUT</p>
                    <h3>Recommendation preview</h3>
                    <p className="muted">
                      Customer-level discount allocation results
                    </p>
                  </div>

                  <span className="result-tag">
                    {recommendations.length} records
                  </span>
                </div>

                {recommendations.length > 0 ? (
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
                        {recommendations.map((row, index) => (
                          <tr key={row.customer_id ?? index}>
                            <td>{row.customer_id ?? "N/A"}</td>
                            <td>
                              {money(row.recommended_discount)}
                            </td>
                            <td>
                              {percent(
                                row.estimated_conversion_probability
                              )}
                            </td>
                            <td>
                              {money(row.expected_discount_spend)}
                            </td>
                            <td>
                              {money(
                                row.estimated_incremental_contribution
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="empty-chart">
                    No recommendation records were returned.
                  </div>
                )}
              </div>

              {optimization.note && (
                <p className="muted small">
                  {optimization.note}
                </p>
              )}

              <div className="synthetic-note">
                Recommendations are generated using synthetic retail
                data and a SciPy Mixed-Integer Linear Programming (MILP)
                optimizer. These are educational estimates and should
                not be used for live pricing decisions without further
                validation.
              </div>

            </div>
          )}
        </section>

        {/* FOOTER */}

        <footer>
          <strong>EconoCausal</strong>
          <span>Educational prototype</span>
          <span>Synthetic retail data</span>
          <span>Not for live pricing decisions</span>
        </footer>

      </main>
    </div>
  );
}
