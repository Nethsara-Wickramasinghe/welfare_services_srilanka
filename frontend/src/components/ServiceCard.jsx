import { useNavigate } from 'react-router-dom';
import '../styles/ServiceCard.css';

export default function ServiceCard({ icon, title, description, link, isImage }) {
  const navigate = useNavigate();

  const handleLearnMore = () => {
    navigate(`/service/${link}`);
  };

  return (
    <div className="service-card">
      <div className="service-icon">
        {isImage ? <img src={icon} alt={title} /> : <span>{icon}</span>}
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      <button className="service-link" onClick={handleLearnMore}>
        Learn More →
      </button>
    </div>
  );
}
