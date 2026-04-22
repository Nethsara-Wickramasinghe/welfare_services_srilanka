import Services from '../components/Services';
import '../styles/ServicesPage.css';

export default function ServicesPage() {
  return (
    <div className="services-page">
      <div className="page-header">
        <h1>Our Services</h1>
        <p>Explore our comprehensive welfare programs designed to support all Sri Lankans</p>
      </div>
      <Services />
    </div>
  );
}
