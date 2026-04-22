import { useEffect, useMemo, useState } from "react";
import "../styles/SWNPage.css";

const API_BASE_URL = "http://127.0.0.1:5000";

const INITIAL_FORM = {
  district_name: "",
  ds_division_name: "",
  gn_division_name: "",
  family_size: "4",
  income_range: "25000-50000",
  employment_status: "employed",
  has_elderly: 0,
  has_disabled_member: 0,
  has_chronic_illness: 0,
  recent_disaster_impact: 0,
  food_insecurity_score: 3,
  healthcare_access_score: 3,
};

const INCOME_OPTIONS = [
  { value: "below-25000", label: "Below Rs. 25,000" },
  { value: "25000-50000", label: "Rs. 25,000 - 50,000" },
  { value: "50000-100000", label: "Rs. 50,000 - 100,000" },
  { value: "100000-200000", label: "Rs. 100,000 - 200,000" },
  { value: "above-200000", label: "Above Rs. 200,000" },
];

const EMPLOYMENT_OPTIONS = [
  { value: "employed", label: "Employed" },
  { value: "self-employed", label: "Self-employed" },
  { value: "unemployed", label: "Unemployed" },
  { value: "retired", label: "Retired" },
  { value: "student", label: "Student" },
  { value: "unable-to-work", label: "Unable to work" },
];

const YES_NO_FIELDS = [
  {
    key: "has_elderly",
    label: "Is there an elderly family member (60+)?",
    description: "Helps identify possible elderly care and home support needs.",
  },
  {
    key: "has_disabled_member",
    label: "Does the household include a disabled family member?",
    description: "Used to surface health and welfare assistance pathways.",
  },
  {
    key: "has_chronic_illness",
    label: "Is anyone managing a chronic illness?",
    description: "Helps identify households that may need health support.",
  },
  {
    key: "recent_disaster_impact",
    label: "Has the household been affected by a recent disaster?",
    description: "Used to identify possible emergency or relief support needs.",
  },
];

function PriorityBadge({ level }) {
  const badgeClass = level ? `priority-badge priority-${level.toLowerCase()}` : "priority-badge";
  return <span className={badgeClass}>{level || "Unknown"}</span>;
}

function AssessmentMethodBadge({ method }) {
  if (!method) return null;

  const isMl = method === "citizen_guidance_ml_v1";
  return (
    <span className={`method-badge ${isMl ? "method-ml" : "method-rules"}`}>
      {isMl ? "Assessment powered by ML model" : "Assessment powered by baseline scoring"}
    </span>
  );
}

function ProgramGroupLabel({ group }) {
  const labels = {
    government_primary: "Government priority programme",
    government_secondary: "Government support pathway",
    ngo_secondary: "NGO or community referral",
  };

  return <span className={`program-group-badge ${group || "default"}`}>{labels[group] || "Support option"}</span>;
}

function OutcomeTypeLabel({ outcomeType }) {
  const labels = {
    direct_welfare_match: "Direct welfare match",
    referral_only_with_service_signal: "Referral path with service signal",
    referral_only: "Referral-only result",
    high_priority_no_programme_match: "High priority without programme match",
    moderate_priority_no_programme_match: "Moderate priority without programme match",
    no_strong_match: "No strong match",
  };

  if (!outcomeType) return null;
  return <span className={`outcome-type-badge ${outcomeType}`}>{labels[outcomeType] || "Guidance result"}</span>;
}

function ProgramOptionSection({ title, emptyCopy, programs }) {
  return (
    <div className="result-card">
      <h3>{title}</h3>
      {programs?.length ? (
        <div className="program-card-list">
          {programs.map((program) => (
            <article key={program.id} className="program-card">
              <div className="program-card-header">
                <div>
                  <ProgramGroupLabel group={program.group} />
                  <h4>{program.display_name}</h4>
                </div>
                {program.official_url ? (
                  <a
                    className="program-link"
                    href={program.official_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Official site
                  </a>
                ) : null}
              </div>

              <p className="program-summary">{program.summary}</p>

              <div className="program-meta">
                <div>
                  <span>Authority</span>
                  <strong>{program.authority}</strong>
                </div>
                {program.contact?.hotline ? (
                  <div>
                    <span>Hotline</span>
                    <strong>{program.contact.hotline}</strong>
                  </div>
                ) : null}
                {program.contact?.phone ? (
                  <div>
                    <span>Phone</span>
                    <strong>{program.contact.phone}</strong>
                  </div>
                ) : null}
                {program.contact?.email ? (
                  <div>
                    <span>Email</span>
                    <strong>{program.contact.email}</strong>
                  </div>
                ) : null}
              </div>

              {program.recommendation_reasons?.length ? (
                <div className="program-section">
                  <h5>Why it was matched</h5>
                  <ul>
                    {program.recommendation_reasons.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {program.documents?.length ? (
                <div className="program-section">
                  <h5>Documents to prepare</h5>
                  <ul>
                    {program.documents.map((document) => (
                      <li key={document}>{document}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {program.next_steps?.length ? (
                <div className="program-section">
                  <h5>What to do for this option</h5>
                  <ol>
                    {program.next_steps.map((step) => (
                      <li key={step}>{step}</li>
                    ))}
                  </ol>
                </div>
              ) : null}
            </article>
          ))}
        </div>
      ) : (
        <p className="result-empty-copy">{emptyCopy}</p>
      )}
    </div>
  );
}

function InlineTipChips({ tips }) {
  if (!tips?.length) return null;

  return (
    <div className="inline-tip-row">
      {tips.map((tip) => (
        <span key={tip} className="inline-tip-chip">{tip}</span>
      ))}
    </div>
  );
}

export default function SocialWelfareNeed() {
  const [locations, setLocations] = useState([]);
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [loadingLocations, setLoadingLocations] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    async function loadLocations() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/locations`);
        if (!response.ok) {
          throw new Error("Could not load official location data.");
        }
        const data = await response.json();
        if (isMounted) {
          setLocations(data.districts || []);
        }
      } catch (err) {
        if (isMounted) {
          setError(err.message || "Failed to load locations.");
        }
      } finally {
        if (isMounted) {
          setLoadingLocations(false);
        }
      }
    }

    loadLocations();
    return () => {
      isMounted = false;
    };
  }, []);

  const selectedDistrict = useMemo(
    () => locations.find((district) => district.value === formData.district_name) || null,
    [locations, formData.district_name]
  );

  const dsOptions = selectedDistrict?.ds_divisions || [];

  const selectedDsDivision = useMemo(
    () => dsOptions.find((ds) => ds.value === formData.ds_division_name) || null,
    [dsOptions, formData.ds_division_name]
  );

  const gnOptions = selectedDsDivision?.gn_divisions || [];
  const employmentTips = useMemo(() => {
    if (formData.employment_status === "self-employed") {
      return ["Self-employed may open livelihood or microfinance referrals."];
    }
    if (formData.employment_status === "unemployed" || formData.employment_status === "unable-to-work") {
      return ["Current employment status may increase income-support or referral matches."];
    }
    if (formData.employment_status === "retired") {
      return ["Retired households may unlock elderly-support guidance when other needs are present."];
    }
    return [];
  }, [formData.employment_status]);

  const householdNeedTips = useMemo(() => {
    const tips = [];
    if (formData.has_elderly === 1) {
      tips.push("Selecting Yes can unlock elderly-support guidance.");
    }
    if (formData.has_disabled_member === 1 || formData.has_chronic_illness === 1) {
      tips.push("Health-related answers can strengthen health-support matches.");
    }
    if (formData.recent_disaster_impact === 1) {
      tips.push("Disaster impact can unlock relief pathways.");
    }
    return tips;
  }, [
    formData.has_chronic_illness,
    formData.has_disabled_member,
    formData.has_elderly,
    formData.recent_disaster_impact,
  ]);

  const difficultyTips = useMemo(() => {
    const tips = [];
    if (formData.food_insecurity_score >= 4) {
      tips.push("Higher food difficulty can strengthen food-support matches.");
    }
    if (formData.healthcare_access_score <= 2) {
      tips.push("Poor healthcare access can strengthen health-support matches.");
    }
    return tips;
  }, [formData.food_insecurity_score, formData.healthcare_access_score]);

  const handleLocationChange = (key, value) => {
    setResult(null);
    setError("");

    setFormData((prev) => {
      if (key === "district_name") {
        return {
          ...prev,
          district_name: value,
          ds_division_name: "",
          gn_division_name: "",
        };
      }

      if (key === "ds_division_name") {
        return {
          ...prev,
          ds_division_name: value,
          gn_division_name: "",
        };
      }

      return {
        ...prev,
        [key]: value,
      };
    });
  };

  const handleValueChange = (key, value) => {
    setResult(null);
    setError("");
    setFormData((prev) => ({ ...prev, [key]: value }));
  };

  const handleFamilySizeChange = (value) => {
    setResult(null);
    setError("");

    if (value === "") {
      setFormData((prev) => ({ ...prev, family_size: "" }));
      return;
    }

    const digitsOnly = value.replace(/\D/g, "");
    const normalizedValue = digitsOnly.replace(/^0+(?=\d)/, "");

    setFormData((prev) => ({ ...prev, family_size: normalizedValue }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setSubmitting(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/api/assess-citizen`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ...formData,
          family_size: Number(formData.family_size),
        }),
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Assessment failed. Please try again.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Could not complete the assessment.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleReset = () => {
    setFormData(INITIAL_FORM);
    setResult(null);
    setError("");
  };

  return (
    <section className="citizen-assessment-page">
      <div className="citizen-assessment-hero">
        <div>
          <span className="eyebrow">Citizen Welfare Guidance</span>
          <h1>Check Your Welfare Support Options</h1>
          <p>
            Use your location and a few household details to get a preliminary
            recommendation on the welfare services most relevant to you, along
            with the likely support priority and next steps.
          </p>
        </div>
        <div className="hero-note-card">
          <h2>How this helps</h2>
          <ul>
            <li>Uses official GN-level population and housing context</li>
            <li>Adapts recommendations to your household situation</li>
            <li>Shows clear next actions instead of just a score</li>
          </ul>
        </div>
      </div>

      <div className="assessment-grid">
        <form className="assessment-panel" onSubmit={handleSubmit}>
          <div className="panel-header">
            <h2>Welfare Guidance Form</h2>
            <p>
              This is a preliminary guidance tool for citizens. Final verification
              should be completed through the appropriate welfare office.
            </p>
          </div>

          {loadingLocations ? (
            <div className="loading-state">Loading official location data...</div>
          ) : (
            <>
              <div className="section-block">
                <h3>1. Select Your Location</h3>
                <div className="form-grid">
                  <label className="field-group">
                    <span>District</span>
                    <select
                      value={formData.district_name}
                      onChange={(event) => handleLocationChange("district_name", event.target.value)}
                      required
                    >
                      <option value="">Select district</option>
                      {locations.map((district) => (
                        <option key={district.value} value={district.value}>
                          {district.name}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="field-group">
                    <span>DS Division</span>
                    <select
                      value={formData.ds_division_name}
                      onChange={(event) => handleLocationChange("ds_division_name", event.target.value)}
                      disabled={!selectedDistrict}
                      required
                    >
                      <option value="">Select DS division</option>
                      {dsOptions.map((ds) => (
                        <option key={ds.value} value={ds.value}>
                          {ds.name}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="field-group field-group-full">
                    <span>GN Division</span>
                    <select
                      value={formData.gn_division_name}
                      onChange={(event) => handleLocationChange("gn_division_name", event.target.value)}
                      disabled={!selectedDsDivision}
                      required
                    >
                      <option value="">Select GN division</option>
                      {gnOptions.map((gn) => (
                        <option key={`${gn.value}-${gn.population_total}`} value={gn.value}>
                          {gn.name}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              </div>

              <div className="section-block">
                <h3>2. Tell Us About Your Household</h3>
                <div className="form-grid">
                  <label className="field-group">
                    <span>Family size</span>
                    <input
                      type="number"
                      min="1"
                      max="20"
                      value={formData.family_size}
                      onChange={(event) => handleFamilySizeChange(event.target.value)}
                      required
                    />
                  </label>

                  <label className="field-group">
                    <span>Monthly income range</span>
                    <select
                      value={formData.income_range}
                      onChange={(event) => handleValueChange("income_range", event.target.value)}
                      required
                    >
                      {INCOME_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="field-group field-group-full">
                    <span>Employment status</span>
                    <select
                      value={formData.employment_status}
                      onChange={(event) => handleValueChange("employment_status", event.target.value)}
                      required
                    >
                      {EMPLOYMENT_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                    <InlineTipChips tips={employmentTips} />
                  </label>
                </div>

                <div className="toggle-grid">
                  {YES_NO_FIELDS.map((field) => (
                    <label key={field.key} className="toggle-card">
                      <div>
                        <span>{field.label}</span>
                        <small>{field.description}</small>
                      </div>
                      <div className="binary-toggle" role="group" aria-label={field.label}>
                        <button
                          type="button"
                          className={formData[field.key] === 0 ? "binary-toggle-option is-active" : "binary-toggle-option"}
                          aria-pressed={formData[field.key] === 0}
                          onClick={() => handleValueChange(field.key, 0)}
                        >
                          No
                        </button>
                        <button
                          type="button"
                          className={formData[field.key] === 1 ? "binary-toggle-option is-active" : "binary-toggle-option"}
                          aria-pressed={formData[field.key] === 1}
                          onClick={() => handleValueChange(field.key, 1)}
                        >
                          Yes
                        </button>
                      </div>
                    </label>
                  ))}
                </div>
                <InlineTipChips tips={householdNeedTips} />
              </div>

              <div className="section-block">
                <h3>3. Current Difficulties</h3>
                <div className="form-grid">
                  <label className="field-group">
                    <span>Food insecurity score</span>
                    <select
                      value={formData.food_insecurity_score}
                      onChange={(event) => handleValueChange("food_insecurity_score", Number(event.target.value))}
                    >
                      <option value={1}>1 - Stable access to food</option>
                      <option value={2}>2 - Mild difficulty</option>
                      <option value={3}>3 - Moderate difficulty</option>
                      <option value={4}>4 - Frequent difficulty</option>
                      <option value={5}>5 - Severe difficulty</option>
                    </select>
                  </label>

                  <label className="field-group">
                    <span>Healthcare access score</span>
                    <select
                      value={formData.healthcare_access_score}
                      onChange={(event) => handleValueChange("healthcare_access_score", Number(event.target.value))}
                    >
                      <option value={1}>1 - Very difficult access</option>
                      <option value={2}>2 - Difficult access</option>
                      <option value={3}>3 - Moderate access</option>
                      <option value={4}>4 - Good access</option>
                      <option value={5}>5 - Very good access</option>
                    </select>
                  </label>
                </div>
                <InlineTipChips tips={difficultyTips} />
              </div>

              {error && <div className="form-error">{error}</div>}

              <div className="form-actions">
                <button type="button" className="secondary-action" onClick={handleReset}>
                  Reset
                </button>
                <button type="submit" className="primary-action" disabled={submitting}>
                  {submitting ? "Assessing..." : "Get Welfare Guidance"}
                </button>
              </div>
            </>
          )}
        </form>

        <div className="assessment-panel result-panel">
          <div className="panel-header">
            <h2>Your Guidance Result</h2>
            <p>
              The result combines your input with official GN-level context to
              give practical, citizen-facing guidance.
            </p>
          </div>

          {!result ? (
            <div className="empty-result">
              <h3>Ready when you are</h3>
              <p>
                Complete the form to see recommended welfare services, your likely
                support priority, and suggested next steps.
              </p>
            </div>
          ) : (
            <div className="result-stack">
              <div className="result-hero">
                <div>
                  <span className="result-label">Estimated priority</span>
                  <PriorityBadge level={result.priority_level} />
                  <AssessmentMethodBadge method={result.assessment_method} />
                  <OutcomeTypeLabel outcomeType={result.outcome_type} />
                </div>
                <div className="result-score-card">
                  <span>Priority confidence</span>
                  <strong>{Math.round((result.priority_score || 0) * 100)}%</strong>
                </div>
              </div>

              {result.result_summary ? (
                <div className="result-card">
                  <h3>Outcome Summary</h3>
                  <p className="result-empty-copy">{result.result_summary}</p>
                </div>
              ) : null}

              <div className="result-guidance-note">
                This is guidance, not final eligibility. Final decisions should be confirmed by the relevant welfare authority.
              </div>

              <div className="result-card">
                <h3>Recommended Services</h3>
                {result.recommended_services?.length ? (
                  <div className="service-pill-list">
                    {result.recommended_services.map((service) => (
                      <div key={service.service} className="service-pill">
                        <span>{service.service}</span>
                        <strong>{Math.round(service.score * 100)}%</strong>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="result-empty-copy">
                    No strongly matched welfare service was identified from the
                    current household and location details.
                  </p>
                )}
              </div>

              <ProgramOptionSection
                title="Direct Welfare Programmes"
                programs={result.direct_welfare_programs}
                emptyCopy="No direct welfare programme match was identified from the current household details."
              />

              <ProgramOptionSection
                title="Relevant Referrals"
                programs={result.referral_programs}
                emptyCopy="No additional referral pathways were identified from the current household details."
              />

              <details className="technical-details">
                <summary>Technical details</summary>

                <div className="result-card technical-card">
                  <h3>Why This Result</h3>
                  {result.household_reasons?.length ? (
                    <>
                      <h4 className="reason-subheading">Household reasons</h4>
                      <ul>
                        {result.household_reasons.map((factor) => (
                          <li key={factor}>{factor}</li>
                        ))}
                      </ul>
                    </>
                  ) : (
                    <p className="result-empty-copy">No household-specific reason was generated for this result.</p>
                  )}

                  {result.location_reasons?.length ? (
                    <>
                      <h4 className="reason-subheading">Location context considered</h4>
                      <ul>
                        {result.location_reasons.map((factor) => (
                          <li key={factor}>{factor}</li>
                        ))}
                      </ul>
                    </>
                  ) : null}
                </div>

                <div className="result-card technical-card">
                  <h3>Location Context</h3>
                  <div className="context-grid">
                    <div>
                      <span>Selected GN</span>
                      <strong>{result.location_context?.gn_division_name}</strong>
                    </div>
                    <div>
                      <span>Population</span>
                      <strong>{result.location_context?.population_total?.toLocaleString?.() || result.location_context?.population_total}</strong>
                    </div>
                    <div>
                      <span>Occupied housing units</span>
                      <strong>{result.location_context?.occupied_housing_units?.toLocaleString?.() || result.location_context?.occupied_housing_units}</strong>
                    </div>
                    <div>
                      <span>Elderly ratio</span>
                      <strong>{result.location_context?.elderly_ratio}</strong>
                    </div>
                  </div>
                </div>
              </details>

              <div className="result-card">
                <h3>What To Do Next</h3>
                <ol>
                  {result.next_steps?.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              </div>

              <div className="result-note">
                This is a preliminary guidance result. Final eligibility and benefit
                decisions should be confirmed by the relevant welfare authority.
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
