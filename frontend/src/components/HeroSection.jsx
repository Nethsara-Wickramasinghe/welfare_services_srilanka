import { useState, useEffect } from 'react';
import '../styles/HeroSection.css';

export default function HeroSection() {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const images = ['/1.jpg', '/2.jpg', '/3.jpg'];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentImageIndex((prevIndex) => (prevIndex + 1) % images.length);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <section className="hero" id="home">
      <div className="hero-content">
        <h1>Welfare Services for All Sri Lankans</h1>
        <p>“A centralized digital platform for social welfare service management in Sri Lanka, enabling efficient service delivery, transparent processes, and secure, easy access for citizens.”</p>
        <div className="hero-buttons">
          <button className="btn btn-primary">Get Started</button>
          <button className="btn btn-secondary">Learn More</button>
        </div>
      </div>
      <div className="hero-image">
        <div className="image-carousel">
          <img 
            src={images[currentImageIndex]} 
            alt="Hero carousel" 
            className="carousel-image"
          />
          <div className="carousel-indicators">
            {images.map((_, index) => (
              <div
                key={index}
                className={`indicator ${index === currentImageIndex ? 'active' : ''}`}
                onClick={() => setCurrentImageIndex(index)}
              ></div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
