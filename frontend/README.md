# Welfare Services Sri Lanka - React Website

A modern, responsive React website built with Vite for government welfare services in Sri Lanka. This project features a component-based architecture that mirrors real-world welfare service websites.

## Project Structure

```
src/
├── components/
│   ├── Navbar.jsx           # Navigation bar with mobile menu
│   ├── HeroSection.jsx      # Hero banner with CTA buttons
│   ├── Services.jsx         # Services grid layout
│   ├── ServiceCard.jsx      # Individual service card component
│   ├── About.jsx            # About section with stats and mission/vision
│   ├── ContactForm.jsx      # Contact form with validation
│   └── Footer.jsx           # Footer with links and info
├── styles/
│   ├── Navbar.css           # Navbar styles
│   ├── HeroSection.css      # Hero section styles
│   ├── Services.css         # Services grid styles
│   ├── ServiceCard.css      # Service card styles
│   ├── About.css            # About section styles
│   ├── ContactForm.css      # Contact form styles
│   └── Footer.css           # Footer styles
├── utils/                   # Utility functions (placeholder)
├── pages/                   # Page components (placeholder for future expansion)
├── App.jsx                  # Main app component
├── App.css                  # Global app styles and button classes
├── index.css                # Global styles and reset
└── main.jsx                 # Entry point
```

## Features

- **Responsive Design**: Mobile-first approach with responsive CSS Grid and Flexbox
- **Component-Based Architecture**: Reusable, maintainable components
- **Modern UI**: Clean, professional design with smooth animations
- **Mobile Menu**: Hamburger menu for mobile devices
- **Contact Form**: Fully functional contact form with validation
- **Services Grid**: Showcase welfare services with icons
- **Statistics Section**: Display key metrics and achievements
- **SEO-Friendly**: Semantic HTML structure

## Components

### Navbar
- Sticky navigation bar with logo
- Mobile hamburger menu
- Navigation links with smooth scrolling

### HeroSection
- Eye-catching banner with main message
- Call-to-action buttons
- Responsive image placeholder

### Services
- Grid layout (6 service cards)
- Service icons and descriptions
- Individual service links

### ServiceCard
- Reusable card component
- Hover effects
- Icon, title, and description

### About
- Company statistics in a grid
- Vision and mission statements
- Responsive stat cards

### ContactForm
- Form with validation
- Success message feedback
- Contact information cards
- Responsive layout

### Footer
- Multiple footer sections
- Quick links
- Social media links
- Copyright information

## Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open [http://localhost:5173](http://localhost:5173) to view the website

## Build for Production

```bash
npm run build
```

This creates an optimized production build in the `dist/` folder.

## Technologies Used

- **React 18** - UI library
- **Vite** - Build tool and dev server
- **CSS3** - Styling with Grid and Flexbox
- **ESLint** - Code quality

## Customization

### Colors
The main color scheme uses `#1a5f7a` (teal) as the primary color. You can update this in the CSS files:
- Primary: `#1a5f7a`
- Accent: `#ffc107` (gold)

### Services
Edit the `services` array in `src/components/Services.jsx` to add or modify services.

### Contact Information
Update contact details in `src/components/ContactForm.jsx` and `src/components/Footer.jsx`.

### Content
Replace placeholder text and images with actual content:
- Update hero section image
- Modify service descriptions
- Update contact information
- Change footer content

## Future Enhancements

- [ ] Add routing with React Router
- [ ] Create dedicated pages for each service
- [ ] Implement backend API integration
- [ ] Add authentication system
- [ ] Create admin dashboard
- [ ] Add blog/news section
- [ ] Multi-language support

## File Descriptions

| File | Purpose |
|------|---------|
| Navbar.jsx | Navigation component with mobile support |
| HeroSection.jsx | Hero banner with promotional content |
| Services.jsx | Services showcase grid |
| ServiceCard.jsx | Reusable service card |
| About.jsx | Company information and stats |
| ContactForm.jsx | Contact form with validation |
| Footer.jsx | Website footer |

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## License

This project is for educational and development purposes.

## Notes

- All image placeholders should be replaced with actual images
- Phone numbers and emails in the contact section are placeholders
- Form submission currently logs to console (integrate with backend API)
- Service icons use Unicode emoji (can be replaced with SVG icons)

---

For more information about customizing or extending this project, refer to the component files and their inline documentation.
