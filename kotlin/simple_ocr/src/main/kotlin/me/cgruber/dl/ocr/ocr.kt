package me.cgruber.dl.ocr

import ArgsParser


class Config(opts: ArgsParser) {

    val foo by opts.opt("--foo", "-f", help="A simple option with a paramter")
    val bar by opts.opt("--bar", "-b", help="An opt with a default value").default { "defaultBar" }
    val baz: Int? by opts.opt("--baz", help="An opt's string converted to an Int", xform = { it.toInt() })
    val bin by opts.opt("--bin", env = "TEST_BINARY")
    val flag by opts.flag("--flag")
    val help by opts.flag("--help", "-h")
}

fun main(vararg args: String) {
    val opts = ArgsParser(*args, "ocr")
    val config = Config(opts)

    if (config.help) {
        println(opts.help())
        System.exit(0)
    }
}