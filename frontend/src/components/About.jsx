import '../styles/About.css';

export default function About() {
  return (
    <section className="about" id="about">
      <div className="about-container">
        <div className="about-content">
          <h2>About Our Welfare Services</h2>
          <p>
            Our mission is to provide comprehensive welfare services and social support to all citizens of Sri Lanka. 
            We are committed to improving the quality of life through various programs and initiatives.
          </p>
          
          <div className="stats-grid">
            <div className="stat-card">
              <h3>500K+</h3>
              <p>Citizens Served</p>
            </div>
            <div className="stat-card">
              <h3>50+</h3>
              <p>Programs</p>
            </div>
            <div className="stat-card">
              <h3>25+</h3>
              <p>Years of Service</p>
            </div>
            <div className="stat-card">
              <h3>100%</h3>
              <p>Commitment</p>
            </div>
          </div>

          <div className="vision-mission">
            <div className="vision">
              <h3>Our Vision</h3>
              <p>A Sri Lanka where every citizen has access to essential welfare services and opportunities for a better life.</p>
            </div>
            <div className="mission">
              <h3>Our Mission</h3>
              <p>To deliver high-quality, accessible welfare services that empower individuals and families to achieve their potential.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
