// import React from 'react'
import './NavBar.css'

const NavBar = () => {
  return (
    <div className='navbar'>
        <ul className="nav-menu">
            <li>Home</li>
            <li>About Me</li>
            <li>Blogs</li>
            <li>Portfolio</li>
            <li>Contact</li>
        </ul>
        <div className="nav-connect">Connect With Me</div>
    </div>
  )
}

export default NavBar