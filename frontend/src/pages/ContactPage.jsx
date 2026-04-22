import ContactForm from '../components/ContactForm';
import '../styles/ContactPage.css';

export default function ContactPage() {
  return (
    <div className="contact-page">
      <div className="page-header">
        <h1>Contact Us</h1>
        <p>We're here to help. Reach out to us with any questions or concerns.</p>
      </div>
      <ContactForm />
      
      <section className="map-section">
        <div className="map-container">
          <h2>Find Us</h2>
          <div className="map-placeholder">
            <div className="map-content">
              <span className="map-icon">📍</span>
              <p>Ministry of Social Services</p>
              <p>Colombo 07, Sri Lanka</p>
              <p className="office-hours">
                <strong>Office Hours:</strong><br />
                Monday - Friday: 8:30 AM - 4:30 PM<br />
                Saturday: 8:30 AM - 12:30 PM
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
