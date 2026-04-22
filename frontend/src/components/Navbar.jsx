import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import '../styles/Navbar.css';

export default function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const closeMenu = () => {
    setIsMenuOpen(false);
  };

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-logo">
          <Link to="/" onClick={closeMenu}>
            <h1>Welfare Services Sri Lanka</h1>
          </Link>
        </div>
        
        <button className="hamburger" onClick={toggleMenu}>
          <span></span>
          <span></span>
          <span></span>
        </button>

        <ul className={`nav-menu ${isMenuOpen ? 'active' : ''}`}>
          <li><Link to="/" className={isActive('/') ? 'active' : ''} onClick={closeMenu}>Home</Link></li>
          <li><Link to="/services" className={isActive('/services') ? 'active' : ''} onClick={closeMenu}>Services</Link></li>
          <li><Link to="/about" className={isActive('/about') ? 'active' : ''} onClick={closeMenu}>About</Link></li>
          <li><Link to="/contact" className={isActive('/contact') ? 'active' : ''} onClick={closeMenu}>Contact</Link></li>
          <li><Link to="/login" className={isActive('/login') ? 'active' : ''} onClick={closeMenu}>Login</Link></li>
        </ul>
      </div>
    </nav>
  );
}
