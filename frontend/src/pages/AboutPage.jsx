import About from '../components/About';
import '../styles/AboutPage.css';

export default function AboutSWPage() {
  return (
    <div className="about-page">
      <div className="page-header">
        <h1>About Us</h1>
        <p>Learn more about our mission to serve the people of Sri Lanka</p>
      </div>
      <About />
      
      <section className="team-section">
        <div className="team-container">
          <h2>Our Leadership Team</h2>
          <div className="team-grid">
            <div className="team-card">
              <div className="team-avatar">👤</div>
              <h3>Dr. Anura Perera</h3>
              <p className="team-role">Director General</p>
              <p>Leading welfare initiatives with over 20 years of experience in public service.</p>
            </div>
            <div className="team-card">
              <div className="team-avatar">👤</div>
              <h3>Mrs. Kumari Silva</h3>
              <p className="team-role">Deputy Director</p>
              <p>Specializing in family support programs and community outreach.</p>
            </div>
            <div className="team-card">
              <div className="team-avatar">👤</div>
              <h3>Mr. Rajan Fernando</h3>
              <p className="team-role">Head of Operations</p>
              <p>Managing nationwide service delivery and program implementation.</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
