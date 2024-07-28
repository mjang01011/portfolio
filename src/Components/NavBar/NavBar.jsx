import "./NavBar.css";

const NavBar = () => {
  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    const headerOffset = 80;
    const elementPosition = element.getBoundingClientRect().top;
    const offsetPosition = elementPosition + window.scrollY - headerOffset;

    window.scrollTo({
      top: offsetPosition,
      behavior: "smooth",
    });
  };

  return (
    <div className="navbar">
      <ul className="nav-menu">
        <li>Home</li>
        <li onClick={() => scrollToSection("about")}>About Me</li>
        {/* <li onClick={() => scrollToSection('blogs')}>Blogs</li> */}
        <li onClick={() => scrollToSection("mywork")}>My Work</li>
        <li onClick={() => scrollToSection("contact")}>Contact</li>
      </ul>
      <div className="nav-connect">Connect With Me</div>
    </div>
  );
};

export default NavBar;
