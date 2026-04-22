import ServiceCard from './ServiceCard';
import '../styles/Services.css';

export default function Services() {
  const services = [
    {
      id: 1,
      icon: 'W',
      title: 'Welfare Guidance',
      description:
        'Get citizen-friendly guidance on relevant welfare services, likely support priority, and next steps based on your location and household situation.',
      link: 'social-welfare-need',
    },
  ];

  return (
    <section className="services" id="services">
      <div className="services-container">
        <h2>Welfare Guidance</h2>
        <p className="section-subtitle">
          A citizen-facing welfare guidance tool that combines your household
          details with official GN-level context.
        </p>

        <div className="services-grid">
          {services.map((service) => (
            <ServiceCard
              key={service.id}
              icon={service.icon}
              title={service.title}
              description={service.description}
              link={service.link}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
